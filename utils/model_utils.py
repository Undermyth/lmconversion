# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import typing
import transformers
import utils
import os
import logging
from transformers.cache_utils import DynamicCache
import sys
import pathlib
import torch.nn.functional as F
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from modeling_gpt_neox import GPTNeoXLayer, GPTNeoXForCausalLM
from modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
OPT_NORM = torch.nn.LayerNorm
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
LLAMA_NORM = transformers.models.llama.modeling_llama.LlamaRMSNorm
CUSTOM_LLAMA_MODEL = LlamaForCausalLM
CUSTOM_LLAMA_LAYER = LlamaDecoderLayer
CUSTOM_LLAMA_NORM = LlamaRMSNorm
MISTRAL_MODEL = transformers.models.mistral.modeling_mistral.MistralForCausalLM
MISTRAL_LAYER = transformers.models.mistral.modeling_mistral.MistralDecoderLayer
MISTRAL_NORM = transformers.models.mistral.modeling_mistral.MistralRMSNorm
QWEN2_MODEL = transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
QWEN2_LAYER = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer
QWEN2_NORM = transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
PYTHIA_MODEL = GPTNeoXForCausalLM
PYTHIA_LAYER = GPTNeoXLayer
PYTHIA_NORM = torch.nn.LayerNorm
INTERNLM2_MODEL = None
INTERNLM2_LAYER = None
INTERNLM2_NORM = None



def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    if isinstance(model, CUSTOM_LLAMA_MODEL):
        return CUSTOM_LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    elif isinstance(model, PYTHIA_MODEL):
        return PYTHIA_MODEL
    elif model.config.architectures[0] == 'InternLM2ForCausalLM':
        global INTERNLM2_MODEL,INTERNLM2_LAYER,INTERNLM2_NORM
        INTERNLM2_MODEL = model.__class__
        INTERNLM2_LAYER = model.model.layers.__class__
        INTERNLM2_NORM = model.model.norm.__class__
        return INTERNLM2_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def get_rope_function_name(model):
    if isinstance(model, (LLAMA_MODEL, CUSTOM_LLAMA_MODEL, MISTRAL_MODEL, QWEN2_MODEL, PYTHIA_MODEL)):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL) or isinstance(model, CUSTOM_LLAMA_MODEL):
        return model.model.layers
    if isinstance(model, MISTRAL_MODEL):
        return model.model.layers
    if isinstance(model, QWEN2_MODEL):
        return model.model.layers
    if isinstance(model, PYTHIA_MODEL):
        return model.model.layers
    if isinstance(model, INTERNLM2_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model



def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_model(
    model_name, hf_token=None
):
    if 'llama' in model_name:
        return get_llama(model_name, hf_token)
    elif 'opt' in model_name:
        return get_opt(model_name)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_type(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, CUSTOM_LLAMA_MODEL):
        return CUSTOM_LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    elif isinstance(model, PYTHIA_MODEL):
        return PYTHIA_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')

def get_norm_type(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_NORM
    elif isinstance(model, CUSTOM_LLAMA_MODEL):
        return CUSTOM_LLAMA_NORM
    elif isinstance(model, OPT_MODEL):
        return OPT_NORM
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_NORM
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_NORM
    elif isinstance(model, PYTHIA_MODEL):
        return PYTHIA_NORM
    elif isinstance(model, INTERNLM2_MODEL):
        return INTERNLM2_NORM
    else:
        raise ValueError(f'Unknown model type {model}')
    
    
    
# def get_embeddings(model, model_type) -> list[torch.nn.Module]:
def get_embeddings(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL or model_type == CUSTOM_LLAMA_MODEL:
        return [model.model.embed_tokens]
    elif model_type == INTERNLM2_MODEL:
        return [model.model.tok_embeddings]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    elif model_type == PYTHIA_MODEL:
        return [model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL or model_type == INTERNLM2_MODEL or model_type == CUSTOM_LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    elif model_type == PYTHIA_MODEL:
        return [layer for layer in model.model.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL or model_type == CUSTOM_LLAMA_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    elif model_type == INTERNLM2_MODEL:
        return model.output
    elif model_type == PYTHIA_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL or model_type == CUSTOM_LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          LLAMA_NORM)
    elif model_type == QWEN2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          QWEN2_NORM)
    elif model_type == MISTRAL_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          MISTRAL_NORM)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    elif model_type == INTERNLM2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          INTERNLM2_NORM)
    elif model_type == PYTHIA_MODEL:
        pre_head_layernorm = model.model.norm
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL or model_type == CUSTOM_LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    else:
        raise ValueError(f'Unknown model type {model_type}')

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.variance_epsilon = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.ones(mean_dim))
        self.use_temporary_parameter = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x.to(input_dtype) * weight


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')

def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type == LLAMA_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'out_proj': [],
            'fc1': [],
            'fc2': []
        }
        captured_outputs = {
            'v_proj': [],
        }
        for name in captured_inputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }


def mv_kv_cache(key_values, model=None, dev=None):
    '''
    move prefixed_key_values to corresponding device through full model or target_dec
    '''
    assert model is None or dev is None
    if key_values is None:
        return None
    key_values = list(key_values)
    if model is not None:
        layers = get_layers(model)
        for layer_index in range(len(key_values)):
            block_dev = next(layers[layer_index].parameters()).device
            key_values[layer_index] = list(key_values[layer_index])
            key_values[layer_index][0] = key_values[layer_index][0].to(block_dev)
            key_values[layer_index][1] = key_values[layer_index][1].to(block_dev)
            key_values[layer_index] = tuple(key_values[layer_index])
            
    if dev is not None:
        for layer_index in range(len(key_values)):
            key_values[layer_index] = list(key_values[layer_index])
            key_values[layer_index][0] = key_values[layer_index][0].to(dev)
            key_values[layer_index][1] = key_values[layer_index][1].to(dev)
            key_values[layer_index] = tuple(key_values[layer_index])
    key_values = tuple(key_values)
    return key_values


def get_kv_cache(prefixed_key_values, bs=1):
    if bs > 1:
        prefixed_key_values = kv_cache_repeat(prefixed_key_values, bs)
    if prefixed_key_values is not None:
        kv_cache = DynamicCache.from_legacy_cache(prefixed_key_values)
    else:
        kv_cache = None
    return kv_cache


def kv_cache_repeat(key_values, bs):
    '''
    bs 1 -> bs n
    '''
    if key_values is None:
        return None
    bs_key_values = {}
    for layer_index in range(len(key_values)):
        bs_key_values[layer_index] = list(key_values[layer_index])
        bs_key_values[layer_index][0] = bs_key_values[layer_index][0].repeat_interleave(bs, dim=0)
        bs_key_values[layer_index][1] = bs_key_values[layer_index][1].repeat_interleave(bs, dim=0)
        bs_key_values[layer_index] = tuple(bs_key_values[layer_index])
    return bs_key_values
    

class WrappedPrefixCausalLM(torch.nn.Module):
    def __init__(self, model, prefixed_key_values):
        super().__init__()
        self.model = model
        self.config = model.config
        self.generation_config = model.generation_config
        self.device = model.device
        self.name_or_path = model.name_or_path
        self.vocab_size = model.vocab_size
        self.prefixed_key_values = prefixed_key_values
        self.bs_prefixed_key_values = prefixed_key_values
    
    def tie_weights(self):
        self.model.tie_weights()

    def forward(self, *args, **kwargs):
        if kwargs.get("past_key_values") is None:
            if len(args) >= 1:
                bs = args[0].shape[0]
            else:
                bs = kwargs["input_ids"].shape[0]
            self.bs_prefixed_key_values = kv_cache_repeat(self.prefixed_key_values, bs)
            kwargs["past_key_values"] = self.bs_prefixed_key_values
        return self.model.forward(*args, **kwargs)

import gc
import torch.nn as nn
from typing import Optional, Tuple
from tqdm import tqdm
from einops import repeat

class PrefixCache:
    def __init__(self, key_value_cache, spike, layer_index) -> None:
        super().__init__()
        self.key_cache, self.value_cache = key_value_cache[layer_index]
        self.spike = spike
        self.T = 8

    def update(self, key_states, value_states, layer_idx, cache_kwargs = None):
        if self.spike:
            if self.key_cache.shape[0] != key_states.shape[0]:
                B = key_states.shape[0] // self.T
                key_cache = repeat(self.key_cache, '... -> B ...', B=B).flatten(0, 1)
                value_cache = repeat(self.value_cache, '... -> B ...', B=B).flatten(0, 1)
                key_cache = repeat(key_cache, '... -> T ...', T=self.T).flatten(0, 1)
                value_cache = repeat(value_cache, '... -> T ...', T=self.T).flatten(0, 1)
            else:
                key_cache = repeat(self.key_cache, '... -> T ...', T=self.T).flatten(0, 1)
                value_cache = repeat(self.value_cache, '... -> T ...', T=self.T).flatten(0, 1)
            # print(key_cache.shape, key_states.shape)
            key_cache = torch.cat([key_cache, key_states], dim=-2)
            value_cache = torch.cat([value_cache, value_states], dim=-2)
            return key_cache, value_cache
        else:
            key_cache = torch.cat([self.key_cache, key_states], dim=-2)
            value_cache = torch.cat([self.value_cache, value_states], dim=-2)
            return key_cache, value_cache

def layerwise_inference(model, spike, prefixed_key_values, dataset, mask, labels=None):
    '''
    model should be on cpu.
    '''
    position_ids = None
    layer_dataset = []

    # catch data in the first layer
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.position_ids = None
        def forward(
            self, 
            hidden_states: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            position_ids: Optional[torch.LongTensor] = None,
            *args, **kwargs
        ):
            layer_dataset.append(hidden_states.detach().cpu())
            self.position_ids = position_ids
            raise ValueError
    
    model.model.embed_tokens.cuda()
    layers = get_layers(model)
    for i in range(len(layers)):
        layers[i].self_attn.past_key_value = PrefixCache(prefixed_key_values, spike=spike, layer_index=i)
    layers[0].cuda()
    layers[0] = Catcher(layers[0])

    for sample in dataset:
        try:
            model(sample.cuda())
        except ValueError:
            pass        
    position_ids = layers[0].position_ids + prefixed_key_values[0][0].shape[-2]

    model.model.embed_tokens.cpu()
    layers[0].cpu()
    torch.cuda.empty_cache()
    layers[0] = layers[0].module

    # layerwise forward
    pbar = tqdm(range(len(layers)), desc='Inferring Layers: ')
    for i in pbar:
        layer = layers[i].cuda()
        for j in range(len(layer_dataset)):
            print(j, len(layer_dataset))
            data = layer_dataset[j].cuda()
            layer_dataset[j] = layer(data, attention_mask=mask, position_ids=position_ids)[0].cpu()
        layers[i].cpu()
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        
    # head: norm and linear
    norm = model.model.norm.cuda()
    head = model.lm_head.cuda()

    for i in range(len(layer_dataset)):
        data = layer_dataset[i].cuda()
        data = norm(data)
        data = head(data)
        layer_dataset[i] = data.cpu()

    model.model.norm.cpu()
    model.lm_head.cpu()
    del norm, head
    gc.collect()
    torch.cuda.empty_cache()

    # labels and losses
    if labels is not None:
        losses = []
        for logits, label in zip(layer_dataset, labels):
            # Shift so that tokens < n predict n
            logits = logits.cuda()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            losses.append(loss.cpu())
        return layer_dataset, losses

    return layer_dataset, None

def layerwise_calibrate(ref_model, model, prefixed_key_values, dataset, mask, labels=None):
    '''
    model should be on cpu.
    '''
    position_ids = None
    layer_dataset = []

    # catch data in the first layer
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.position_ids = None
        def forward(
            self, 
            hidden_states: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            position_ids: Optional[torch.LongTensor] = None,
            *args, **kwargs
        ):
            layer_dataset.append(hidden_states.detach().cpu())
            self.position_ids = position_ids
            raise ValueError
        
    # preparation of reference model and its input data
    ref_model.model.embed_tokens.cuda()
    ref_layers = get_layers(ref_model)
    for i in range(len(ref_layers)):
        ref_layers[i].self_attn.past_key_value = PrefixCache(prefixed_key_values, spike=False, layer_index=i)
    ref_layers[0].cuda()
    ref_layers[0] = Catcher(ref_layers[0])
    for sample in dataset:
        try: 
            ref_model(sample.cuda())
        except ValueError:
            pass

    ref_position_ids = ref_layers[0].position_ids + prefixed_key_values[0][0].shape[-2]
    ref_model.model.embed_tokens.cpu()
    ref_layers[0].cpu()
    torch.cuda.empty_cache()
    ref_layers[0] = ref_layers[0].module
    ref_layer_dataset = layer_dataset
    layer_dataset = []
    
    # preparation of target model and its input data
    model.model.embed_tokens.cuda()
    layers = get_layers(model)
    for i in range(len(layers)):
        layers[i].self_attn.past_key_value = PrefixCache(prefixed_key_values, spike=True, layer_index=i)
    layers[0].cuda()
    layers[0] = Catcher(layers[0])

    for sample in dataset:
        try:
            model(sample.cuda())
        except ValueError:
            pass        
    position_ids = layers[0].position_ids + prefixed_key_values[0][0].shape[-2]

    model.model.embed_tokens.cpu()
    layers[0].cpu()
    torch.cuda.empty_cache()
    layers[0] = layers[0].module

    # layerwise calibration
    ref_layer_output = []
    pbar = tqdm(range(len(layers)), desc='Calibrating Layers: ')
    for i in pbar:

        # produce the target from ref model
        with torch.no_grad():
            ref_layer = ref_layers[i].cuda()
            for j in range(len(ref_layer_dataset)):
                data = ref_layer_dataset[j].cuda()
                ref_layer_output.append(ref_layer(data, attention_mask=mask, position_ids=ref_position_ids)[0].detach().cpu())
            ref_layers[i].cpu()
            del ref_layer
            gc.collect()
            torch.cuda.empty_cache()
            print('ref layer done')

        # optimize the target model
        layer = layers[i].cuda()
        params = []
        params.extend(layer.input_layernorm.rsqrtop.approximator.parameters())
        params.extend(layer.post_attention_layernorm.rsqrtop.approximator.parameters())
        optimizer = torch.optim.SGD(params, lr=1e-5)
        for j in range(len(layer_dataset)):
            optimizer.zero_grad()
            data = layer_dataset[j].cuda().detach()
            pred = layer(data, attention_mask=mask, position_ids=position_ids)[0]
            loss = F.huber_loss(pred, ref_layer_output[j].cuda().detach(), reduction='mean')
            try:
                loss.backward()
            except:
                pass
            optimizer.step()
            print(f'layer {i} sample {j} loss: {loss.item()}')

        # forward again to get the corrected output as the input to the next layer
        for j in range(len(layer_dataset)):
            data = layer_dataset[j].cuda()
            layer_dataset[j] = layer(data, attention_mask=mask, position_ids=position_ids)[0].cpu()
        layers[i].cpu()
        del layer
        gc.collect()
        torch.cuda.empty_cache()

        ref_layer_dataset = ref_layer_output
        ref_layer_output = []