"""Qwen Phase 包装与 checkpoint 神经元结构创建。"""
# Extracted/adapted from the local PhaseSNN implementation; see NOTICE.
import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP, Qwen2RMSNorm, Qwen2SdpaAttention,
)
from phase.phase_layer import (
    phaseSnnEmbedding, phaseSnnIdentity, phaseSnnLinear, phaseSnnLinear2,
    snnQwen2MLP, snnQwen2RMSNorm, snnQwenAttention,
)
from phase.phase_neuron import FSNeuron
from utils.hadamard_utils import get_hadK


def set_module(model, name, module):
    parent_name, _, child_name = name.rpartition('.')
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def wrap_to_phase_model(model, T=1):
    """保留原 checkpoint 的模块路径和外层 T=1 计算布局。"""
    for name, module in list(model.named_modules()):
        replacement = None
        if name == 'lm_head':
            replacement = phaseSnnLinear2(module, T=T)
        elif isinstance(module, nn.Embedding):
            replacement = phaseSnnEmbedding(module, T=T)
        elif isinstance(module, Qwen2RMSNorm):
            replacement = snnQwen2RMSNorm(module, T=T)
        elif isinstance(module, Qwen2SdpaAttention):
            replacement = snnQwenAttention(module, module.config, T)
        elif isinstance(module, Qwen2MLP):
            replacement = snnQwen2MLP(module, module.config, T)
        if replacement is not None:
            set_module(model, name, replacement)


def register_online_had(model):
    for name, module in model.named_modules():
        if isinstance(module, phaseSnnLinear) and name.endswith('down_proj'):
            module.had_K, module.K = get_hadK(model.config.intermediate_size)
            module.online_full_had = True
            module.fp32_had = False


def init_phase_neurons(model, timestep):
    """创建默认单粒度神经元占位结构，所有参数随后必须从 checkpoint 恢复。"""
    def neuron(width):
        return FSNeuron(
            T=timestep,
            quantized_shape=(1, width),
            quantized_item_stat=torch.ones(width),
        )

    for name, module in list(model.named_modules()):
        if isinstance(module, phaseSnnLinear):
            layer_name = name.rsplit('.', 1)[-1]
            if layer_name not in {'q_proj', 'k_proj', 'v_proj', 'up_proj', 'gate_proj'}:
                module.input_quantizer = neuron(module.in_features)
                module.use_act_quant = True
        elif isinstance(module, snnQwen2RMSNorm):
            module.output_quantizer = neuron(module.out_features)
            module.use_act_quant = True
        elif isinstance(module, phaseSnnIdentity):
            module.input_quantizer = neuron(module.out_features)
            module.use_act_quant = True

    return {
        f'{name}.{parameter}'
        for name, module in model.named_modules()
        if isinstance(module, FSNeuron)
        for parameter in ('d', 'tau', 'h', 'theta', 'v0')
    }
