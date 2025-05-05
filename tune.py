import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, register_online_had, init_k_quantizer, init_v_quantizer, init_q_quantizer, init_a_quantizer, apply_naive_sdpa
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from utils.train_utils import load_json_as_namespace,create_logger
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from accelerate import dispatch_model
from utils.data_utils import layerwise_calibration

from spike.ops import SpikeSoftmax, SpikeQuantizer, SpikeRMSNorm, SpikeLlamaMLP
from spike.netfit import NonLinearOp
from spike.spike_utils import firing_pre_hook, avg_after_hook, avg_after_tuple_hook, firing_after_hook
import functools

from functools import partial

torch.backends.cudnn.benchmark = True

def backward_debug_hook(name, module, grad_input, grad_output):
    print(f"Backward pass completed for {module.__class__.__name__}:{name}!")

def calibrate(ref_model, model, tokenizer, prefixed_key_values, args, logger):
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map, skip_keys='past_key_values') # set skip_keys to avoid a bug
    prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, model)
    results_str=""
    if args.eval_ppl:
        datasets = ["wikitext2"]
        ppl_results = layerwise_calibration(args, ref_model, model, tokenizer, prefixed_key_values, datasets)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            results_str += f"{ppl_results[dataset]:.2f} "


def get_pretrain_model(empty_model, quant_config, args, spike, logger):
    model = copy.deepcopy(empty_model)
    # wrap rope for online_had and rope output capture
    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn, 
                    rope_function_name, 
                    config=model.config,
                    online_had=quant_config.qk_online_had)   

    # init weight quantizer
    if quant_config.wbits < 16:
        logger.info('init weight quantizer')
        init_weight_quantizer(quant_config, model, minmax_init=False)

    # init input quantizer
    if quant_config.input_bits < 16:
        logger.info('init input quantizer')
        init_input_quantizer(quant_config, model,  minmax_init=False)

    # init kv quantizer
    if quant_config.v_bits < 16:
        logger.info('init v quantizer')
        init_v_quantizer(quant_config, model,  minmax_init=False)

    # if True:
    if quant_config.k_bits < 16:
        # consistently init for wrap rope 
        logger.info('init k quantizer')
        init_k_quantizer(quant_config, model,  minmax_init=False)

    # [added] init Q quantizer
    if quant_config.q_bits < 16:
        logger.info('init q quantizer')
        init_q_quantizer(quant_config, model,  minmax_init=False)

    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=args.quant_model_path,device_map=device_map,dtype=torch.float16)
    model.cpu()

    # 1. spiking matmul (attention)
    layers = model_utils.get_layers(model)
    for layer in layers:
        apply_naive_sdpa(layer.self_attn, spike=spike, T=args.T)

    if spike:
        hooks = []
        layers = model_utils.get_layers(model)
        prehook = functools.partial(firing_pre_hook, T=args.T)
        afterhook = functools.partial(firing_after_hook, T=args.T)
        avghook = functools.partial(avg_after_hook, T=args.T)
        avgtuplehook = functools.partial(avg_after_tuple_hook, T=args.T)

        # 2. spiking softmax, q, k
        for layer in layers:
            q_quantizer = layer.self_attn.apply_rotary_pos_emb_qk_rotation_wrapper.q_quantizer
            k_quantizer = layer.self_attn.apply_rotary_pos_emb_qk_rotation_wrapper.k_quantizer
            
            layer.self_attn.apply_rotary_pos_emb_qk_rotation_wrapper.q_quantizer = SpikeQuantizer.from_pretrained(q_quantizer, args.T)
            layer.self_attn.apply_rotary_pos_emb_qk_rotation_wrapper.k_quantizer = SpikeQuantizer.from_pretrained(k_quantizer, args.T)
            layer.self_attn.softmax = SpikeSoftmax('spike/exp.pth', 'spike/inv.pth', args.T)
        
        # 3. spiking rmsnorm
        acts = torch.load('rinv_input_acts.pth', weights_only=True)
        maxs = np.array(list(map(lambda x: 1. / math.sqrt(x.min()), acts)))
        mins = np.array(list(map(lambda x: 1. / math.sqrt(x.max()), acts)))
        alphas = (maxs - mins) / args.rmsnorm_range
        alpha_iter = iter(alphas)
        for layer in layers:
            alpha = next(alpha_iter)
            layer.input_layernorm = SpikeRMSNorm(layer.input_layernorm.variance_epsilon, alpha, 'spike/rinv.pth', args.T)
            alpha = next(alpha_iter)
            layer.post_attention_layernorm = SpikeRMSNorm(layer.post_attention_layernorm.variance_epsilon, alpha, 'spike/rinv.pth', args.T)

        # 4. spiking mlp
        for layer in layers:
            layer.mlp.act_fn = NonLinearOp.from_pretrained('spike/silu_new.pth')
            layer.mlp.down_proj.input_quantizer = SpikeQuantizer.from_pretrained(layer.mlp.down_proj.input_quantizer, args.T)

        for layer in layers:

            hooks.append(layer.input_layernorm.register_forward_pre_hook(prehook))
            hooks.append(layer.input_layernorm.register_forward_hook(avghook))

            hooks.append(layer.input_layernorm.register_forward_hook(afterhook))
            hooks.append(layer.self_attn.register_forward_hook(avgtuplehook))

            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(prehook))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(avghook))

            hooks.append(layer.mlp.register_forward_pre_hook(prehook))
            hooks.append(layer.mlp.register_forward_hook(avghook))

            # for name, module in layer.named_modules():
            #     hooks.append(module.register_full_backward_hook(lambda module, grad_input, grad_output: backward_debug_hook(name, module, grad_input, grad_output)))

    return model

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
    parser.add_argument("--output_dir", default="./log/test", type=str, help="direction of logging file")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--ppl_seqlen", type=int, default=256, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")
    parser.add_argument("--T", type=int, default=8, help="time step")
    parser.add_argument("--rmsnorm_range", type=float, default=10., help="rmsnorm range")


    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)

    quant_config = load_json_as_namespace(os.path.join(args.quant_model_path, 'prefixequant_config.json'))
    # if quant_config['set_prefixed_tokens']:
    if quant_config.set_prefixed_tokens:
        prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
    else:
        prefixed_key_values = None


    # init quantized model
    config = AutoConfig.from_pretrained(args.quant_model_path,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.quant_model_path, use_fast=True,legacy=False,trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(args.quant_model_path, config=config, device_map='cpu',torch_dtype=torch.float16,trust_remote_code=True)
    wrap_to_quant_model(model)
    # register on-line hadadamrd transformation
    if quant_config.down_online_had:
        register_online_had(model)
    
    ref_model = get_pretrain_model(model, quant_config, args, False, logger)
    model = get_pretrain_model(model, quant_config, args, True, logger)
    ref_model.half()
    model.half()
    calibrate(ref_model, model, tokenizer, prefixed_key_values, args, logger)

    # import ipdb
    # ipdb.set_trace()

if __name__ == "__main__":
    print(sys.argv)
    main()
