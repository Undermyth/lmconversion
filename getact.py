import os
import sys
import random
import numpy as np
import torch
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, register_online_had, init_k_quantizer, init_v_quantizer, init_q_quantizer, init_a_quantizer, apply_naive_sdpa
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from main import evaluate
from utils.train_utils import load_json_as_namespace,create_logger
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model

from quantize.quant_norm import QuantRMSNorm

torch.backends.cudnn.benchmark = True



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
    parser.add_argument("--output_dir", default="./log/test", type=str, help="direction of logging file")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--ppl_seqlen", type=int, default=512, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")
    parser.add_argument("--T", type=int, default=8, help="time step")

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
    tokenizer = AutoTokenizer.from_pretrained(args.quant_model_path, use_fast=False,legacy=False,trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(args.quant_model_path, config=config, device_map='cpu',torch_dtype=torch.float16,trust_remote_code=True)
    wrap_to_quant_model(model)
    # register on-line hadadamrd transformation
    if quant_config.down_online_had:
        register_online_had(model)
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

    if quant_config.a_bits < 16:
        logger.info('init a quantizer')
        init_a_quantizer(quant_config, model,  minmax_init=False)

    # model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=args.quant_model_path,device_map=device_map,dtype=torch.float16)
    
    class NormHook:
        def __init__(self):
            self.tensor_list = []
        def __call__(self, module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            self.tensor_list.append(input.pow(2).mean(dim=-1) + 1e-5)
            return output
    
    hooks = []
    handlers = []
    for name, module in model.named_modules():
        if isinstance(module, QuantRMSNorm):
            hook = NormHook()
            hooks.append(hook)
            handlers.append(module.register_forward_hook(hook))

    model.half()    # to make sure same evaluation results with main
    evaluate(model, tokenizer, prefixed_key_values,  args,logger)

    for handler in handlers:
        handler.remove()
    
    acts = []
    for hook in hooks:
        tensor = torch.cat(hook.tensor_list, dim=0).cpu()
        acts.append(tensor)

    # print(acts[16])
    # acts = torch.cat(acts, dim=0).cpu()
    # print(acts.shape)

    torch.save(acts, 'acts.pt')

    # import ipdb
    # ipdb.set_trace()

if __name__ == "__main__":
    print(sys.argv)
    main()
