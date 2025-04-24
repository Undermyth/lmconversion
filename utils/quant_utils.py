from collections import OrderedDict
from quantize.int_linear_fake import QuantLinear
import quantize.int_linear_fake as int_linear_fake
from quantize.quant_norm import QuantRMSNorm
from utils import model_utils
from utils.model_utils import RMSN,get_kv_cache
import utils.rotation_utils as rotation_utils
from utils.rotation_utils import QKRotationWrapper
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import torch
from torch import nn
from typing  import Optional
from quantize.quantizer import UniformAffineQuantizer, SoftmaxQuantizer
import utils.hadamard_utils as hadamard_utils
import functools
from tqdm import tqdm
from transformers.models.llama.modeling_llama import repeat_kv
import math

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from modeling_gpt_neox import GPTNeoXLayer

def get_act_stat(model, dataloader, accumulate_type='max', prefixed_tokens=None, online_had=False):
    model.eval()
    num_heads = model.config.num_attention_heads
    if hasattr(model.config, 'num_key_value_heads'):
        num_kv_heads = model.config.num_key_value_heads
    else:
        num_kv_heads = num_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    device = next(model.parameters()).device
    act_stat = {}
    prefixed_length = len(prefixed_tokens) if prefixed_tokens is not None else 0

    if online_had:
        had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)

    def stat_tensor(name, tensor, type):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        ema_factor = 0.99
        if accumulate_type == 'max':
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        elif accumulate_type == 'mean':
            comming_max = torch.mean(tensor, dim=0).float().cpu()
        key_name = f"{name}.{type}"
        if key_name in act_stat:
            act_stat[key_name] = ema_factor * act_stat[key_name] + (1-ema_factor) * comming_max
        else:
            act_stat[key_name] = comming_max

    def stat_input_hook(m, x, y, name):
        if 'apply_rotary_pos_emb_qk_rotation_wrapper' in name:
            input_Q = x[0].transpose(1, 2).flatten(-2)
            input_K = x[1].transpose(1, 2).flatten(-2)
            output_Q = y[0].transpose(1, 2).flatten(-2)
            output_K = y[1].transpose(1, 2).flatten(-2) # [comment] reshape into B, L, (H*d)
            if prefixed_length > 0:
                input_Q = input_Q[:,prefixed_length:, ]
                input_K = input_K[:,prefixed_length:, ]
                output_Q = output_Q[:,prefixed_length:, ]
                output_K = output_K[:,prefixed_length:, ]
            stat_tensor(name, input_Q, 'input_Q')
            stat_tensor(name, input_K, 'input_K')
            stat_tensor(name, output_Q, 'output_Q')
            stat_tensor(name, output_K, 'output_K')
        elif 'softmax' in name:
            if isinstance(x, tuple):
                x = x[0]
            x = x.transpose(-1, -3).contiguous()
            y = y.transpose(-1, -3).contiguous()
            stat_tensor(name, x, 'input')
            stat_tensor(name, y, 'output')
        else:
            if isinstance(x, tuple):
                x = x[0]
            if prefixed_length > 0:
                x_ = x[:,prefixed_length:, ]
                y_ = y[:,prefixed_length:, ]
            else:
                x_,y_ = x, y
            if online_had and 'down_proj' in name:
                x_ = hadamard_utils.matmul_hadU_cuda(x_, had_K, K)
            stat_tensor(name, x_, 'input')
            stat_tensor(name, y_, 'output')

    hooks = []
    for name, m in model.named_modules():
        # if isinstance(m, nn.Linear):
        if isinstance(m, (nn.Linear,LlamaRMSNorm,RMSN,QuantLinear,QuantRMSNorm,QKRotationWrapper,nn.Softmax)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(len(dataloader)), desc='obtain activation stat'):
        data = dataloader[i][0]
        if prefixed_tokens is not None:
            data = torch.cat([torch.tensor([prefixed_tokens]),data],dim=1)
        model(data.to(device))

    for h in hooks:
        h.remove()

    return act_stat


def wrap_to_quant_model(model):
    '''
    replace nn.Linear and norm layer to correspond quantization counterparts
    '''
    for name, module in model.named_modules():
        # skip lm_head quantization
        if 'lm_head' in name or 'embed_out' in name:
            continue
        # skip quantization of norm for lm_head
        if 'model.norm' in name or 'final_layer_norm' in name:
            continue
        if isinstance(module,torch.nn.Linear):
            quantlinear = int_linear_fake.QuantLinear(module)
            set_op_by_name(model, name, quantlinear)  
            del module  
        elif isinstance(module,(RMSN, LlamaRMSNorm)):
            quantnorm = QuantRMSNorm(module)
            set_op_by_name(model, name, quantnorm)  
            del module 
            

def register_online_had(model):
    for name, module in model.named_modules():
        if isinstance(module,int_linear_fake.QuantLinear) and 'down_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            module.online_full_had = True
            module.had_K = had_K
            module.K = K
            module.fp32_had = False

def init_weight_quantizer(args, model, minmax_init=True):
    for name, module in model.named_modules():
        if isinstance(module,int_linear_fake.QuantLinear):
            # layer_name = name.split('.')[-1]
            # wbits = args.special_w_quant_bit if layer_name in args.special_w_quant_layer else args.wbits
            wbits = args.wbits
            module.wbits = wbits
            if wbits >= 16:
                continue
            w_group_size=args.w_group_size
            w_asym=args.w_asym
            quantized_item_stat=module.weight if minmax_init else None
            module.use_weight_quant = True
            module.weight_quantizer = UniformAffineQuantizer(wbits, module.weight.shape,  w_asym, w_group_size,
                                                           quantized_item_stat=quantized_item_stat,
                                                           quant_type='weight',
                                                           minmax_init=minmax_init)
            sym_stat = "asymmetric" if w_asym else 'symmetric'
            print(f'weight quantization: set {name} as w{wbits}g{w_group_size} {sym_stat} quantization')

def init_input_quantizer(args, model, activation_stat=None, minmax_init=True):
    for name, module in model.named_modules():
        # skip lm_head quantization
        if isinstance(module,int_linear_fake.QuantLinear):
            # skip quant at norm layer
            layer_name = name.split('.')[-1]
            if layer_name in ['q_proj','k_proj','v_proj','up_proj','gate_proj'] or 'lm_head' in name:
                continue
            input_bits = args.input_bits
            module.input_bits = input_bits
            if input_bits >= 16:
                continue
            input_group_size = args.input_group_size
            input_asym = args.input_asym
            input_mode = args.input_mode
            input_stat = activation_stat[f'{name}.input'] if activation_stat is not None else None
            activation_clipping=args.activation_clipping
            module.use_act_quant = True
            quantized_shape = (1,module.in_features)
            module.input_quantizer = UniformAffineQuantizer(input_bits, quantized_shape, input_asym, input_group_size,
                                                            quantized_item_stat=input_stat,
                                                            quant_type='activation',
                                                            mode=input_mode, 
                                                            activation_clipping=activation_clipping,
                                                            minmax_init=minmax_init)
            sym_stat = "asymmetric" if input_asym else 'symmetric'
            print(f'input activation quantization: set {name} as {input_bits}-bit {input_group_size} groupsize {input_mode} {sym_stat} quantization')
        elif isinstance(module,(QuantRMSNorm)):
            # quantization for the input of q_proj/k_proj/v_porj/up_proj/gate_proj are putted in normalization layer
            output_bits = args.input_bits
            if output_bits >= 16:
                continue
            module.output_bits = output_bits
            output_group_size = args.input_group_size
            output_asym = args.input_asym
            output_mode = args.input_mode
            output_stat = activation_stat[f'{name}.output'] if activation_stat is not None else None
            module.use_act_quant = True
            activation_clipping=args.activation_clipping
            quantized_shape = (1,module.out_features)
            module.output_quantizer =  UniformAffineQuantizer(output_bits, quantized_shape, output_asym, output_group_size,
                                                            quantized_item_stat=output_stat,
                                                            quant_type='activation',
                                                            mode=output_mode, 
                                                            activation_clipping=activation_clipping,
                                                            minmax_init=minmax_init)
            sym_stat = "asymmetric" if output_asym else 'symmetric'
            print(f'output activation quantization: set {name} as {output_bits}-bit {output_group_size} groupsize {output_mode} {sym_stat} quantization')



def init_v_quantizer(args, model, activation_stat=None, minmax_init=True):
    # for the quantization of k/v output (kv-cache quantization)
    for name, module in model.named_modules():
        if isinstance(module,int_linear_fake.QuantLinear) and 'v_proj' in name:
            output_bits = args.v_bits
            module.output_bits = output_bits
            if output_bits >= 16:
                continue
            output_group_size = args.kv_group_size
            output_asym = args.kv_asym
            output_mode = args.kv_mode
            output_stat = activation_stat[f'{name}.output'] if activation_stat is not None else None
            module.use_act_quant = True
            activation_clipping=args.activation_clipping
            quantized_shape = (1,module.out_features)
            module.output_quantizer = UniformAffineQuantizer(output_bits, quantized_shape, output_asym, output_group_size, 
                                                            quantized_item_stat=output_stat,
                                                            quant_type='activation',
                                                            mode=output_mode,
                                                            minmax_init=minmax_init,
                                                            activation_clipping=activation_clipping)
            sym_stat = "asymmetric" if output_asym else 'symmetric'
            print(f'v-cache quantization: set {name} as {output_bits}-bit {output_group_size} groupsize {output_mode} {sym_stat} quantization')


    
    
def init_k_quantizer(args, model, activation_stat=None, minmax_init=True):
    num_heads = model.config.num_attention_heads
    # num_kv_heads = model.config.num_key_value_heads
    if hasattr(model.config, 'num_key_value_heads'):
        num_kv_heads = model.config.num_key_value_heads
    else:
        num_kv_heads = num_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    assert args.kv_group_size in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
    # for the quantization of k/v output (kv-cache quantization)
    if args.k_pre_rope:
        for name, module in model.named_modules():
            if isinstance(module,int_linear_fake.QuantLinear) and 'k_proj' in name:
                output_bits = args.k_bits
                module.output_bits = output_bits
                if output_bits >= 16:
                    continue
                output_group_size = args.kv_group_size
                output_asym = args.kv_asym
                output_mode = args.kv_mode
                output_stat = activation_stat[f'{name}.output'] if activation_stat is not None else None
                module.use_act_quant = True
                quantized_shape = (1,module.out_features)
                module.output_quantizer = UniformAffineQuantizer(output_bits, quantized_shape, output_asym, output_group_size,
                                                                quantized_item_stat=output_stat,
                                                                quant_type='activation',
                                                                mode=output_mode,
                                                                minmax_init=minmax_init)
                sym_stat = "asymmetric" if output_asym else 'symmetric'
                print(f'k-cache quantization: set {name} as {output_bits}-bit {output_group_size} groupsize {output_mode} {sym_stat} quantization')
    else:
        for name, module in model.named_modules():
            if isinstance(module, rotation_utils.QKRotationWrapper):
                output_bits = args.k_bits
                if output_bits >= 16:
                    continue
                output_group_size = args.kv_group_size
                output_asym = args.kv_asym
                output_mode = args.kv_mode
                output_stat = activation_stat[f'{name}.output_K'] if activation_stat is not None else None
                module.use_k_quant = True
                module.k_bits = output_bits
                module.online_had = args.qk_online_had
                quantized_shape = (1,kv_dim)
                module.k_quantizer = UniformAffineQuantizer(output_bits, quantized_shape, output_asym, output_group_size,
                                                                quantized_item_stat=output_stat,
                                                                quant_type='activation',
                                                                mode=output_mode,
                                                                minmax_init=minmax_init)
                sym_stat = "asymmetric" if output_asym else 'symmetric'
                print(f'k-cache quantization: set {name} as {output_bits}-bit {output_group_size} groupsize {output_mode} {sym_stat} quantization')

# [added]
def init_q_quantizer(args, model, activation_stat=None, minmax_init=True):
    model_dim = model.config.hidden_size
    for name, module in model.named_modules():
        if isinstance(module, rotation_utils.QKRotationWrapper):
            output_bits = args.q_bits
            if output_bits >= 16:
                continue
            output_stat = activation_stat[f'{name}.output_Q'] if activation_stat is not None else None
            module.use_q_quant = True
            module.q_bits = output_bits
            quantized_shape = (1,model_dim)
            output_asym = args.kv_asym # [warn] use kv_asym for q/k/v
            output_mode = args.kv_mode # [warn] use kv_mode for q/k/v
            output_group_size = args.kv_group_size # [warn] by default per-tensor
            module.q_quantizer = UniformAffineQuantizer(output_bits, quantized_shape, output_asym, output_group_size,
                                                                quantized_item_stat=output_stat,
                                                                quant_type='activation',
                                                                mode=output_mode,
                                                                minmax_init=minmax_init)
            sym_stat = "asymmetric" if output_asym else 'symmetric'
            print(f'q quantization: set {name} as {output_bits}-bit {output_group_size} groupsize {output_mode} {sym_stat} quantization')


# [added]
def init_a_quantizer(args, model, activation_stat=None, minmax_init=True):
    model_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    layers = model_utils.get_layers(model)

    layer_list = [layer for layer in layers]
    softmax_list = [name for name, module in model.named_modules() if isinstance(module, nn.Softmax)]

    for layer, name in zip(layer_list, softmax_list):
        module = layer.self_attn
        output_bits = args.a_bits
        if output_bits >= 16:
            continue
        output_stat = activation_stat[f'{name}.output'] if activation_stat is not None else None
        module.use_act_quant = True
        module.a_bits = output_bits
        quantized_shape = (1,num_heads)
        output_asym = args.kv_asym # [warn] use kv_asym for q/k/v
        output_mode = args.kv_mode # [warn] use kv_mode for q/k/v
        output_group_size = 1
        module.a_quantizer = SoftmaxQuantizer(output_bits, quantized_shape, output_asym, output_group_size,
                                                            quantized_item_stat=output_stat,
                                                            quant_type='activation',
                                                            mode=output_mode,
                                                            minmax_init=minmax_init)
        sym_stat = "asymmetric" if output_asym else 'symmetric'
        print(f'attn quantization: set {name} as {output_bits}-bit {output_group_size} groupsize {output_mode} {sym_stat} quantization')

            
       
@torch.no_grad()
def weight_layer_mse_init(module, input_feat, n_grid=20, max_shrink=0.5):
# def weight_layer_mse_init(module, input_feat, n_grid=50, max_shrink=0.75):
    '''
    inspired by https://github.com/mit-han-lab/llm-awq/blob/3665e1abbf04139aa037254ff4ff3f261bd68a40/awq/quantize/auto_clip.py#L11
    only for weight
    '''
    quantizer = module.weight_quantizer
    w = module.weight
    dev = w.device
    # original_scale = quantizer.scale.clone().to(dev)       # init with MAX-MIN
    group_size = quantizer.group_size
    if isinstance(input_feat,list):
        input_feat = torch.cat(input_feat, dim=0)
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(input_feat.shape[0], -1, group_size)

    out_c = w.shape[0]
    # oc_batch_size = 256 if out_c % 256 == 0 else 64  # prevent OOM
    oc_batch_size = 64 if out_c % 256 == 0 else 32  # prevent OOM
    assert out_c % oc_batch_size == 0

    w =  w.reshape(out_c, -1, group_size)

    ori_scale = quantizer.scale.clone()

    input_feat = input_feat.to(w.device)
    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    
    best_scale_list = []
    min_errs_list = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w_sub = w[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]
        best_scale = ori_scale[i_b*quantizer.inc_groups*oc_batch_size:(i_b+1)*quantizer.inc_groups*oc_batch_size].clone()
        if quantizer.zero_point is not None:
            zero_point = quantizer.zero_point[i_b*quantizer.inc_groups*oc_batch_size:(i_b+1)*quantizer.inc_groups*oc_batch_size]
        else:
            zero_point = None
        org_out = torch.einsum('ngc, gcm -> ngm', input_feat.to(w_sub.dtype), w_sub.permute(1,2,0))
        min_errs = torch.ones_like(best_scale) * 1e9
        for i_s in range(int(max_shrink * n_grid)):
            clip_factor = (1 - i_s / n_grid)
            cur_scale = best_scale * clip_factor
            w_sub = w_sub.view(-1, group_size)
            q_w = quantizer.custom_quant(w_sub, cur_scale, zero_point)
            q_w = q_w.view(oc_batch_size, -1, group_size)
            cur_out = torch.einsum('ngc, gcm -> ngm', input_feat.to(w_sub.dtype), q_w.permute(1,2,0))

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=0).view(min_errs.shape)
            del q_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_scale[cur_best_idx] = cur_scale[cur_best_idx]
        best_scale_list.append(best_scale)
        min_errs_list.append(min_errs)

    best_scale = torch.cat(best_scale_list, dim=0)
    quantizer.scale.data = best_scale
    min_errs = torch.cat(min_errs_list, dim=0)
    return min_errs.mean()

@torch.no_grad()
def k_cache_mse_init(module, kv_cache, softmax=False, n_grid=20, max_shrink=0.5):
    quantizer = module.k_quantizer
    num_key_value_groups = module.num_key_value_groups
    num_key_value_heads = module.num_key_value_heads
    head_dim = module.head_dim,
    group_size = quantizer.group_size
    assert group_size == head_dim[0]
    dev = quantizer.scale.device
    Q_list = []
    K_list = []
    for Q,K in kv_cache:
        Q_list.append(Q)
        K_list.append(K)
    Q = torch.cat(Q_list, dim=0)
    K = torch.cat(K_list, dim=0)
    (bsz, num_heads, seq_len, head_dim) = K.shape

    # head-wise init to avoid OOM
    best_scale_list = []
    min_errs_list = []
    ori_scale = quantizer.scale.clone().to(dev)
    for head_index in range(num_heads):
        K_sub = K[:, head_index:head_index+1, ]
        Q_sub = Q[:, head_index*num_key_value_groups:(head_index+1)*num_key_value_groups, ]
        best_scale = ori_scale[head_index:head_index+1]
        rep_K = repeat_kv(K_sub, num_key_value_groups)
        org_out = torch.matmul(Q_sub, rep_K.transpose(2, 3)) / math.sqrt(head_dim)
        if softmax:
            org_out = nn.functional.softmax(org_out, dim=-1, dtype=torch.float32)
        if quantizer.zero_point is not None:
            zero_point = quantizer.zero_point[head_index:head_index+1]
        else:
            zero_point = None
        min_errs = torch.ones_like(best_scale) * 1e9
        for i_s in range(int(max_shrink * n_grid)):
            clip_factor = (1 - i_s / n_grid)
            cur_scale = best_scale * clip_factor
            q_K = K_sub.transpose(1, 2).flatten(-2)
            q_K = quantizer.custom_quant(q_K, cur_scale, zero_point).reshape((bsz, seq_len, 1, head_dim)).transpose(1, 2).to(Q)
            cur_out = torch.matmul(Q_sub, repeat_kv(q_K, num_key_value_groups).transpose(2, 3))/math.sqrt(head_dim)
            if softmax:
                cur_out = nn.functional.softmax(cur_out, dim=-1, dtype=torch.float32)
            err = (cur_out - org_out).pow(2)
            # [comment] actually supports only per-tensor scale. err can't be reshaped to account for oer channel losses
            err = err.reshape((bsz,1,num_key_value_groups,seq_len,seq_len)).mean(dim=(0,2,3,4)).view(min_errs.shape)
            del q_K
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx].to(min_errs)
            best_scale[cur_best_idx] = cur_scale[cur_best_idx]
        best_scale_list.append(best_scale)
        min_errs_list.append(min_errs)

    best_scale = torch.cat(best_scale_list, dim=0)
    quantizer.scale.data = best_scale
    min_errs = torch.cat(min_errs_list, dim=0)
    return min_errs.mean()

# [added]
def qk_mse_init(module: QKRotationWrapper, kv_cache, softmax=False, n_grid=20, max_shrink=0.5):
    q_quantizer = module.q_quantizer
    k_quantizer = module.k_quantizer
    num_key_value_groups = module.num_key_value_groups
    num_key_value_heads = module.num_key_value_heads
    head_dim = module.head_dim
    # group_size = quantizer.group_size
    # assert group_size == head_dim[0]
    dev = q_quantizer.scale.device
    Q_list = []
    K_list = []
    for Q,K in kv_cache:
        Q_list.append(Q)
        K_list.append(K)
    Q = torch.cat(Q_list, dim=0)
    K = torch.cat(K_list, dim=0)
    (bsz, num_heads, seq_len, head_dim) = K.shape

    # head-wise init to avoid OOM
    q_best_scale_list = []
    k_best_scale_list = []
    min_errs_list = []
    q_ori_scale = q_quantizer.scale.clone().to(dev)
    k_ori_scale = k_quantizer.scale.clone().to(dev)
    assert q_ori_scale.shape == k_ori_scale.shape
    for head_index in range(num_heads):
        K_sub = K[:, head_index:head_index+1, ]
        Q_sub = Q[:, head_index*num_key_value_groups:(head_index+1)*num_key_value_groups, ]
        q_best_scale = q_ori_scale[head_index:head_index+1]
        k_best_scale = k_ori_scale[head_index:head_index+1]
        rep_K = repeat_kv(K_sub, num_key_value_groups)
        org_out = torch.matmul(Q_sub, rep_K.transpose(2, 3)) / math.sqrt(head_dim)
        if softmax:
            org_out = nn.functional.softmax(org_out, dim=-1, dtype=torch.float32)
        if q_quantizer.zero_point is not None:
            q_zero_point = q_quantizer.zero_point[head_index:head_index+1]
        else:
            q_zero_point = None
        if k_quantizer.zero_point is not None:
            k_zero_point = k_quantizer.zero_point[head_index:head_index+1]
        else:
            k_zero_point = None
        min_errs = torch.ones_like(q_best_scale) * 1e9

        for q_i in range(int(max_shrink * n_grid)):
            for k_i in range(int(max_shrink * n_grid)):
                q_clip_factor = (1 - q_i / n_grid)
                k_clip_factor = (1 - k_i / n_grid)
                q_cur_scale = q_best_scale * q_clip_factor
                k_cur_scale = k_best_scale * k_clip_factor
                quant_q = Q_sub.transpose(1, 2).flatten(-2)
                quant_k = K_sub.transpose(1, 2).flatten(-2)
                quant_q = q_quantizer.custom_quant(quant_q, q_cur_scale, q_zero_point).reshape((bsz, seq_len, num_key_value_groups, head_dim)).transpose(1, 2).to(Q)
                quant_k = k_quantizer.custom_quant(quant_k, k_cur_scale, k_zero_point).reshape((bsz, seq_len, 1, head_dim)).transpose(1, 2).to(K)
                cur_out = torch.matmul(quant_q, repeat_kv(quant_k, num_key_value_groups).transpose(2, 3))/math.sqrt(head_dim)
                if softmax:
                    cur_out = nn.functional.softmax(cur_out, dim=-1, dtype=torch.float32)
                err = (cur_out - org_out).pow(2)
                err = err.reshape((bsz,1,num_key_value_groups,seq_len,seq_len)).mean(dim=(0,2,3,4)).view(min_errs.shape)
                del quant_q
                del quant_k
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx].to(min_errs)
                q_best_scale[cur_best_idx] = q_cur_scale[cur_best_idx]
                k_best_scale[cur_best_idx] = k_cur_scale[cur_best_idx]
        q_best_scale_list.append(q_best_scale)
        k_best_scale_list.append(k_best_scale)
        min_errs_list.append(min_errs)

    # [comment] what about zero point? not initialized?
    q_best_scale = torch.cat(q_best_scale_list, dim=0)
    q_quantizer.scale.data = q_best_scale
    k_best_scale = torch.cat(k_best_scale_list, dim=0)
    k_quantizer.scale.data = k_best_scale
    min_errs = torch.cat(min_errs_list, dim=0)

    return min_errs.mean()

# [added]
def attn_mse_init_static(quantizer, data, n_grid=20, max_shrink=0.5):
    dev = quantizer.scale.device
    original_scale = quantizer.scale.clone().to(dev)       # init with MAX-MIN
    best_scale = quantizer.scale.clone().to(dev)
    min_errs = torch.ones_like(best_scale) * 1e9
    if isinstance(data,list):
        data = torch.cat(data, dim=0)
    bs, h, seq_len, _ = data.shape
    for i_s in range(int(max_shrink * n_grid)):
        clip_factor = (1 - i_s / n_grid)
        cur_scale = original_scale * clip_factor
        quantizer.scale.data = cur_scale
        q_data = quantizer(data)
        err = (data - q_data).pow(2).mean(dim=(0,2,3)).view(min_errs.shape)
        del q_data
        cur_best_idx = err < min_errs
        min_errs[cur_best_idx] = err[cur_best_idx].to(min_errs)
        best_scale[cur_best_idx] = cur_scale[cur_best_idx]

    quantizer.scale.data = best_scale
    return min_errs.mean()


@torch.no_grad()
def tensor_mse_init_static(quantizer, data, n_grid=20, max_shrink=0.5):
    dev = quantizer.scale.device
    group_size = quantizer.group_size
    original_scale = quantizer.scale.clone().to(dev)       # init with MAX-MIN
    best_scale = quantizer.scale.clone().to(dev)
    min_errs = torch.ones_like(best_scale) * 1e9
    if isinstance(data,list):
        data = torch.cat(data, dim=0)
    bs, seq_len, dims = data.shape
    for i_s in range(int(max_shrink * n_grid)):
        clip_factor = (1 - i_s / n_grid)
        cur_scale = original_scale * clip_factor
        quantizer.scale.data = cur_scale
        q_data = quantizer(data)
        err = (data - q_data).pow(2).reshape((bs, seq_len, -1, group_size)).mean(dim=(0,1,3)).view(min_errs.shape)
        del q_data
        cur_best_idx = err < min_errs
        min_errs[cur_best_idx] = err[cur_best_idx].to(min_errs)
        best_scale[cur_best_idx] = cur_scale[cur_best_idx]

    quantizer.scale.data = best_scale
    return min_errs.mean()

# @torch.no_grad()
# def tensor_mse_init_dynamic(quantizer, data):
#     dev = quantizer.scale.device
#     group_size = quantizer.group_size
#     min_errs = torch.inf
#     if isinstance(data,list):
#         data = torch.cat(data, dim=0)
#     bs, seq_len, dims = data.shape
#     if quantizer.asym:
#         searched_upbound_factors = torch.linspace(0.6, 1, 9).to(dev)
#         searched_lowbound_factors = torch.linspace(0.6, 1, 9).to(dev)
#         for upbound_fator in searched_upbound_factors:
#             for lowbound_factor in searched_lowbound_factors:
#                 quantizer.upbound_factor.data = upbound_fator
#                 quantizer.lowbound_factor.data = lowbound_factor        
#         q_data = quantizer(data)
#         err = (data - q_data).pow(2).reshape((bs, seq_len, -1, group_size)).mean(dim=(0,1,3)).view(min_errs.shape)
#         del q_data
#         cur_best_idx = err < min_errs
#         min_errs[cur_best_idx] = err[cur_best_idx].to(min_errs)
#         best_scale[cur_best_idx] = cur_scale[cur_best_idx]

#     quantizer.scale.data = best_scale
#     return min_errs.mean()




@torch.no_grad()
def block_mse_init_static(quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, attention_mask, position_ids):
    '''
    share a clipping scale accross a layer
    '''
    loss_func = torch.nn.MSELoss()
    best_loss = torch.inf
    best_scale = None
    original_scale = quantizer.scale.clone().to(dev)
    # coarse_search_factors = torch.linspace(0.45, 0.95, 8).to(dev)
    # coarse_search_factors = torch.linspace(0.40, 0.95, 8).to(dev)
    coarse_search_factors = torch.linspace(0.25, 0.95, 8).to(dev)
    bs = data_inputs.shape[0]
    for clip_factor in coarse_search_factors:
        cur_scale = original_scale * clip_factor
        quantizer.scale.data = cur_scale
        past_key_value = get_kv_cache(prefixed_key_values,bs)
        with torch.cuda.amp.autocast():
            quant_out = qblock(data_inputs,attention_mask=attention_mask, position_ids=position_ids,past_key_value=past_key_value)[0]
        cur_loss = loss_func(quant_out,data_gts)
        if cur_loss <= best_loss:
            best_loss = cur_loss
            best_scale = cur_scale
            best_clip_factor = clip_factor
    
    fine_search_factors = torch.linspace(best_clip_factor-0.05, best_clip_factor+0.05, 10).to(dev)

    for clip_factor in fine_search_factors:
        cur_scale = original_scale * clip_factor
        quantizer.scale.data = cur_scale
        past_key_value = get_kv_cache(prefixed_key_values,bs)
        with torch.cuda.amp.autocast():
            quant_out = qblock(data_inputs,position_ids=position_ids,past_key_value=past_key_value)[0]
        cur_loss = loss_func(quant_out,data_gts)
        if cur_loss <= best_loss:
            best_loss = cur_loss
            best_scale = cur_scale
            best_clip_factor = clip_factor

    quantizer.scale.data = best_scale
    return best_clip_factor, best_loss

@torch.no_grad()
def block_mse_init_dynamic(quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, attention_mask, position_ids):
    '''
    search for the clipping value of dynamic quantizer
    '''
    loss_func = torch.nn.MSELoss()
    best_loss = torch.inf
    bs = data_inputs.shape[0]
    
    if quantizer.asym:
        searched_upbound_factors = torch.linspace(0.6, 1, 9).to(dev)
        searched_lowbound_factors = torch.linspace(0.6, 1, 9).to(dev)
        for upbound_fator in searched_upbound_factors:
            for lowbound_factor in searched_lowbound_factors:
                quantizer.upbound_factor.data = upbound_fator
                quantizer.lowbound_factor.data = lowbound_factor
                past_key_value = get_kv_cache(prefixed_key_values,bs)
                with torch.cuda.amp.autocast():
                    quant_out = qblock(data_inputs,attention_mask=attention_mask, position_ids=position_ids,past_key_value=past_key_value)[0]
                cur_loss = loss_func(quant_out,data_gts)
                if cur_loss <= best_loss:
                    best_loss = cur_loss
                    best_upbound_factor = upbound_fator
                    best_lowbound_factor = lowbound_factor
        quantizer.upbound_factor.data = best_upbound_factor
        quantizer.lowbound_factor.data = best_lowbound_factor
        best_clip_factor = (best_lowbound_factor.item(),best_upbound_factor.item())
    else:
        searched_bound_factors = torch.linspace(0.6, 1, 9).to(dev)
        for bound_factor in searched_bound_factors:
            quantizer.bound_factor.data = bound_factor
            past_key_value = get_kv_cache(prefixed_key_values,bs)
            with torch.cuda.amp.autocast():
                if isinstance(qblock, GPTNeoXLayer):
                    quant_out = qblock(data_inputs,attention_mask=attention_mask, position_ids=position_ids,layer_past=past_key_value)[0]
                else:
                    quant_out = qblock(data_inputs,attention_mask=attention_mask, position_ids=position_ids,past_key_value=past_key_value)[0]
            cur_loss = loss_func(quant_out,data_gts)
            # print(cur_loss)
            # import pdb;pdb.set_trace()
            if cur_loss <= best_loss:
                best_loss = cur_loss
                best_bound_factor = bound_factor
        quantizer.bound_factor.data = best_bound_factor
        best_clip_factor = best_bound_factor.item()
    return best_clip_factor, best_loss

@torch.no_grad()
def mse_init(qblock, prefixed_key_values, dev, data_inputs, attention_mask, position_ids, logger, args, data_gt_asym=None):
# def mse_init(qblock, prefixed_key_values, dev, data_inputs, position_ids, logger, args):
    # part1: obtain the intermediate input/output of each linear layer
    set_quant_state(qblock,weight_quant=False,act_quant=False)
    input_activation_dict = {}
    output_activation_dict = {}
    def get_activation_hook(layer_name, input_activation_dict, output_activation_dict):
        def hook(model, input, output):
            if not 'qk_rotation_wrapper' in layer_name:
                input = input[0]

            if layer_name in input_activation_dict:
                input_activation_dict[layer_name].append(input)
            else:
                input_activation_dict[layer_name] = [input]
            if layer_name in output_activation_dict:
                output_activation_dict[layer_name].append(output)
            else:
                output_activation_dict[layer_name] = [output]
        return hook
    hooks = []
    for name, layer in qblock.named_modules():
        if isinstance(layer, (int_linear_fake.QuantLinear, QKRotationWrapper, nn.Softmax)):
            hooks.append(layer.register_forward_hook(get_activation_hook(name, input_activation_dict, output_activation_dict)))
    data_gts = torch.zeros_like(data_inputs)
    for i in range(len(data_inputs)):
        with torch.cuda.amp.autocast():
            # data_gts[i:i+1] = qblock(data_inputs[i:i+1],position_ids=position_ids,past_key_value=get_kv_cache(prefixed_key_values))[0]    
            data_gts[i:i+1] = qblock(data_inputs[i:i+1],attention_mask=attention_mask, position_ids=position_ids,past_key_value=get_kv_cache(prefixed_key_values))[0]    
    if data_gt_asym is not None:
        data_gts = data_gt_asym
    for h in hooks:
        h.remove()
    # end of part 1
    # part2: mse init of quantizer
    batch_attention_mask = None if attention_mask is None else attention_mask.repeat(data_gts.shape[0],1,1,1) 
    for name, module in qblock.named_modules():

        if isinstance(module, nn.Softmax):
            # init a quantizer
            if hasattr(qblock.self_attn, 'a_quantizer') and qblock.self_attn.a_quantizer.n_bits < 16:
                qblock.self_attn.a_quantizer.activate()
                best_loss = attn_mse_init_static(qblock.self_attn.a_quantizer, output_activation_dict[name])
                logger.info(f"[{name}_attn_quantizer] best_loss:{best_loss} ")

        if isinstance(module, (int_linear_fake.QuantLinear,QuantRMSNorm,QKRotationWrapper)):
            module.set_quant_state(weight_quant=False,act_quant=True)
            # init input quantizer
            if hasattr(module,'input_quantizer')  and module.input_quantizer.n_bits<16:
                module.input_quantizer.activate()
                if module.input_quantizer.mode=='static':
                    best_clip_factor, best_loss = block_mse_init_static(module.input_quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, batch_attention_mask, position_ids)
                    logger.info(f"[{name}_input_quantizer] clipping factor: ({best_clip_factor}); best_loss:{best_loss} ")
                elif module.input_quantizer.mode=='dynamic' and module.input_quantizer.activation_clipping:
                    best_clip_factor, best_loss = block_mse_init_dynamic(module.input_quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, batch_attention_mask, position_ids)
                    logger.info(f"[{name}_input_quantizer] clipping factor: ({best_clip_factor}); best_loss:{best_loss} ")
            
            # init output quantizer        
            if hasattr(module, 'output_quantizer') and module.output_quantizer.n_bits<16:
                module.output_quantizer.activate()
                # V cache quantization
                if module.output_quantizer.mode=='static':
                    if 'v_proj' in name and module.output_quantizer.inc_groups > 1:
                        best_loss = tensor_mse_init_static(module.output_quantizer,output_activation_dict[name])
                        logger.info(f"[{name}_output_quantizer] best_loss:{best_loss} ")
                    else:
                        best_clip_factor, best_loss = block_mse_init_static(module.output_quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, batch_attention_mask, position_ids)
                        logger.info(f"[{name}_output_quantizer] clipping factor: ({best_clip_factor}); best_loss:{best_loss} ")
                elif module.output_quantizer.mode=='dynamic' and module.output_quantizer.activation_clipping:
                    # if 'v_proj' in name and module.output_quantizer.inc_groups > 1:
                    #     pass
                    # else:
                    #     best_clip_factor, best_loss = block_mse_init_dynamic(module.output_quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, batch_attention_mask, position_ids)
                    #     logger.info(f"[{name}_output_quantizer] clipping factor: ({best_clip_factor}); best_loss:{best_loss} ")
                    best_clip_factor, best_loss = block_mse_init_dynamic(module.output_quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, batch_attention_mask, position_ids)
                    logger.info(f"[{name}_output_quantizer] clipping factor: ({best_clip_factor}); best_loss:{best_loss} ")
            
            # init qk quantizer
            if hasattr(module, 'q_quantizer') and module.q_quantizer.mode=='static' and module.q_quantizer.n_bits<16 and \
               hasattr(module, 'k_quantizer') and module.k_quantizer.mode=='static' and module.k_quantizer.n_bits<16:
                module.q_quantizer.activate()
                module.k_quantizer.activate()
                if module.q_quantizer.inc_groups > 1 and module.k_quantizer.inc_groups > 1:
                    best_loss = qk_mse_init(module,output_activation_dict[name], softmax=True)
                    logger.info(f"[{name}_qk_quantizer] best_loss:{best_loss} ")

            # init k quantizer
            elif hasattr(module,'k_quantizer') and module.k_quantizer.mode=='static' and module.k_quantizer.n_bits<16:
                module.k_quantizer.activate()
                if module.k_quantizer.inc_groups > 1:
                    best_loss = k_cache_mse_init(module,output_activation_dict[name], softmax=True)
                    logger.info(f"[{name}_k_quantizer] best_loss:{best_loss} ")
                else:
                    best_clip_factor, best_loss = block_mse_init_static(module.k_quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, batch_attention_mask, position_ids)
                    logger.info(f"[{name}_k_quantizer] clipping factor: ({best_clip_factor:.2f}); best_loss:{best_loss} ")
            module.set_quant_state(weight_quant=True,act_quant=False)

            # init weight quantizer
            if hasattr(module,'weight_quantizer') and module.weight_quantizer.n_bits<16:
                module.weight_quantizer.activate()
                if 'q_proj' in name or 'k_proj' in name:
                    if args.skip_qk_weight_init:
                        continue
                    elif args.block_qk_weight_init:
                        best_clip_factor, best_loss = block_mse_init_static(module.weight_quantizer,qblock, prefixed_key_values, dev, data_inputs, data_gts, batch_attention_mask, position_ids)
                        logger.info(f"[{name}_weight_quantizer] clipping factor: ({best_clip_factor:.2f}); best_loss:{best_loss} ")
                    else:
                        best_loss = weight_layer_mse_init(module, input_activation_dict[name])
                        logger.info(f"[{name}_weight_quantizer] best_loss:{best_loss} ")
                else:
                        best_loss = weight_layer_mse_init(module, input_activation_dict[name])
                        logger.info(f"[{name}_weight_quantizer] best_loss:{best_loss} ")
            module.set_quant_state(weight_quant=False,act_quant=False)
    # end of part 2
            
            
            
 

class MultiBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_list = nn.ModuleList([])
    
    def add_block(self, block):
        self.block_list.append(block)
        
    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None):
        for block in self.block_list:
            hidden_states = block(hidden_states, attention_mask=attention_mask,position_ids=position_ids)[0]
        return (hidden_states, )


def set_weight_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        # if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
        if n.find('weight') > -1 and not (n.find('weight_quantizer') > -1):
            m.requires_grad = requires_grad
    return iter(params)

def weight_parameters(model):
    params = []
    for n, m in model.named_parameters():
        # if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
        if n.find('weight') > -1 and not (n.find('weight_quantizer') > -1):
            params.append(m)
    return iter(params)
    
def set_scale_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find('in_scale') > -1 or n.find('out_scale') > -1:
            m.requires_grad = requires_grad
    return iter(params)

def scale_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('in_scale') > -1 or n.find('out_scale') > -1:
            params.append(m)
    return iter(params)

def set_quant_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        # if (n.find('scale') > -1 or n.find('zero_point') > -1) and (not n.find('smooth_scale') > -1):
        if (n.find('scale') > -1 or n.find('zero_point') > -1 or n.find('bound_factor') > -1)  and (not (n.find('in_scale') > -1 or n.find('out_scale') > -1)):
            m.requires_grad = requires_grad
    return iter(params)  

def quant_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if (n.find('scale') > -1 or n.find('zero_point') > -1 or n.find('bound_factor') > -1) and (not (n.find('in_scale') > -1 or n.find('out_scale') > -1)):
            params.append(m)
    return iter(params)  


def trainable_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if m.requires_grad:
            params.append(m)
    return iter(params)  

def trainable_parameters_num(model):
    params = []
    total = 0
    for n, m in model.named_parameters():
        if m.requires_grad:
            total += m.numel()
    return total

def set_quant_state(model, weight_quant: bool = False, act_quant: bool = False):
    for m in model.modules():
        # if isinstance(m, QuantLinear):
        if isinstance(m, (QuantLinear,QuantRMSNorm,QKRotationWrapper)):
            m.set_quant_state(weight_quant, act_quant)

def activate_quantizer(model):
    for m in model.modules():
        if isinstance(m, UniformAffineQuantizer):
            m.activate()

def deactivate_quantizer(model):
    for m in model.modules():
        if isinstance(m, UniformAffineQuantizer):
            m.deactivate()
            
@torch.no_grad()   
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if module.wbits < 16:
                module.weight.data = module.weight_quantizer(module.weight.data)


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     


def get_named_linears(module, type):
    # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)
        


def combine_linear_layers(*linears):
    if len(linears) < 2:
        raise ValueError("at least two linear layers")
    
    in_features = linears[0].in_features
    for linear in linears:
        assert linear.in_features == in_features, "input dims must be the same"
    
    combined_out_features = sum(linear.out_features for linear in linears)
    
    combined_linear = nn.Linear(in_features, combined_out_features,bias=None)
    
    # combine the original weights
    with torch.no_grad():
        start = 0
        for linear in linears:
            end = start + linear.out_features
            combined_linear.weight[start:end, :] = linear.weight
            # combined_linear.bias[start:end] = linear.bias
            start = end
    
    return combined_linear  


def check_quantizer(model):
    for name, module in model.named_modules():
        if isinstance(module, UniformAffineQuantizer):
            bits = module.n_bits
            sym = 'asymmetric' if module.asym else 'symmetric'
            if module.enable:
                print(f'{name}: {bits}-bit {sym} quantization')
            

def get_quant_config(args):
    quantization_config = {}
    quantization_config["wbits"] = args.wbits
    quantization_config["w_group_size"] = args.w_group_size
    quantization_config["w_asym"] = args.w_asym
    quantization_config["input_bits"] = args.input_bits
    quantization_config["input_group_size"] = args.input_group_size
    quantization_config["input_asym"] = args.input_asym
    quantization_config["input_mode"] = args.input_mode
    quantization_config["q_bits"] = args.q_bits # [added]
    quantization_config["k_bits"] = args.k_bits
    quantization_config["v_bits"] = args.v_bits
    quantization_config["a_bits"] = args.a_bits
    quantization_config["kv_group_size"] = args.kv_group_size
    quantization_config["kv_asym"] = args.kv_asym
    quantization_config["k_pre_rope"] = args.k_pre_rope
    quantization_config["kv_mode"] = args.kv_mode
    quantization_config["down_online_had"] = args.down_online_had and args.pre_rotate
    quantization_config["qk_online_had"] = args.qk_online_had and args.pre_rotate
    quantization_config["real_quant"] = args.real_quant    
    quantization_config["set_prefixed_tokens"] = args.set_prefixed_tokens    
    quantization_config["activation_clipping"] = args.activation_clipping    
    return quantization_config

from quantize.quantizer import round_ste, UniformAffineQuantizer
import pathlib
import sys
path = pathlib.Path(__file__).parent.parent
sys.path.append(str(path))
from spike.ops import spike_matmul, spike_matmul_mean
from spike.triton_ops import spike_matmul_triton, spike_matmul_mean_triton
import functools
from einops import rearrange, repeat
# [added]
# Efficient implementation equivalent to the following:
# @torch.jit.script
def sdpa_wrapper(spike=False, T=None):
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        if spike:
            assert T is not None
            B, H, _, _ = query.shape
            query = query.flatten(0, 1)
            key = key.flatten(0, 1).transpose(-2, -1)
            attn_weight = spike_matmul_mean(query, key, T) * scale_factor
            attn_weight = rearrange(attn_weight, '(B H) ... -> B H ...', B=B, H=H)
        else:
            attn_weight = query @ key.transpose(-2, -1) * scale_factor

        attn_weight += attn_bias
        attn_weight = self.softmax(attn_weight)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        if hasattr(self, "a_quantizer") and hasattr(self, "a_bits") and self.a_bits < 16:
            attn_weight = self.a_quantizer(attn_weight)
        
        if spike:
            value = value.flatten(0, 1)
            attn_weight = attn_weight.flatten(0, 1)
            res = spike_matmul(attn_weight, value, T)
            res = rearrange(res, '(B H) ... -> B H ...', B=B, H=H)
            return res
        else:
            res = attn_weight @ value
            return res
    return scaled_dot_product_attention

def apply_naive_sdpa(module, spike=False, T=None):
    '''
    module should be self_attn (LlamaAttention)
    '''
    module.softmax = nn.Softmax(dim=-1)
    forword_func = module.forward
    new_sdpa_func = sdpa_wrapper(spike=spike, T=T)
    def wrapped_forward(self, *args, **kwargs):
        temp = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = new_sdpa_func.__get__(self)
        res = forword_func(*args, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = temp
        return res
    module.forward = wrapped_forward.__get__(module)