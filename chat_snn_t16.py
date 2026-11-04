# Extracted from the local PhaseSNN implementation; see NOTICE and LICENSE.
"""从本地 T=16 Phase-SNN Qwen checkpoint 进行多轮聊天。"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map, load_checkpoint_in_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    GenerationConfig,
)

from utils.runtime import load_json_as_namespace, mv_kv_cache, replicate_past_key_values
from phase.phase_util import (
    init_phase_neurons,
    register_online_had,
    wrap_to_phase_model,
)

EXPECTED_TIMESTEP = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用本地 T=16 Phase-SNN Qwen checkpoint 进行多轮对话"
    )
    parser.add_argument("--model-path", required=True, help="保存模型的本地目录，包含权重、tokenizer、Phase 配置与前缀 KV")
    parser.add_argument("--T", "--timestep", dest="timestep", type=int, default=16)
    parser.add_argument("--max-memory", default="20GiB")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--system-prompt", default="你是一个有帮助的中文助手。")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def require_file(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"缺少{label}: {path}")


def load_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-SNN 推理需要 CUDA GPU")

    model_path = Path(args.model_path).expanduser().resolve()
    require_file(model_path / "config.json", "模型配置")
    require_file(model_path / "prefixequant_config.json", "Phase 配置")
    spike_config = load_json_as_namespace(
        str(model_path / "prefixequant_config.json")
    )
    timestep = int(spike_config.T)
    if timestep != EXPECTED_TIMESTEP:
        raise ValueError(
            f"该程序要求 T={EXPECTED_TIMESTEP}，但模型配置为 T={timestep}"
        )
    if args.timestep != timestep:
        raise ValueError(
            f"命令行指定 T={args.timestep}，但模型配置为 T={timestep}"
        )

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    if config.model_type != "qwen2":
        raise ValueError("仅支持 Qwen2 架构的 Phase checkpoint")
    if getattr(spike_config, "wbits", 16) != 16 or getattr(spike_config, "real_quant", False):
        raise ValueError("此推理入口仅支持原始 T=16 浮点权重 Phase checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=False, legacy=False, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # checkpoint 已完成离线旋转，不能再次调用 rotate_model。
    # 第一次加载恢复标准 Qwen 权重；Phase neuron 层在包装后创建。
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        device_map="cpu",
        torch_dtype=torch.float16,
        local_files_only=True,
        output_loading_info=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    wrap_to_phase_model(model, T=1)
    if bool(getattr(spike_config, "down_online_had", False)):
        register_online_had(model)
    neuron_keys = init_phase_neurons(model, timestep)
    missing_neuron_keys = neuron_keys - set(loading_info["unexpected_keys"])
    if missing_neuron_keys:
        raise ValueError(f"checkpoint 缺少 Phase 神经元参数（不能使用普通 Qwen 权重）: {sorted(missing_neuron_keys)[:8]}")

    block_name = model.model.layers[0].__class__.__name__
    max_memory = {
        index: args.max_memory for index in range(torch.cuda.device_count())
    }
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=[block_name],
    )

    # Standard Qwen loading ignores neuron keys, so restore again after wrapping.
    load_checkpoint_in_model(
        model,
        checkpoint=str(model_path),
        device_map=device_map,
        dtype=torch.float16,
    )
    model = dispatch_model(
        model, device_map=device_map, skip_keys="past_key_values"
    )
    model.half().eval()

    generation_path = model_path / "generation_config.json"
    if generation_path.is_file():
        model.generation_config = GenerationConfig.from_pretrained(str(model_path), local_files_only=True)

    prefix_cache = None
    if bool(getattr(spike_config, "set_prefixed_tokens", False)):
        prefix_path = model_path / "prefixed_key_values.pth"
        require_file(prefix_path, "前缀 KV")
        ann_cache = torch.load(
            str(prefix_path), map_location="cpu", weights_only=True
        )
        prefix_cache = replicate_past_key_values(ann_cache, T=1)
        prefix_cache = mv_kv_cache(prefix_cache, model=model)

    input_device = next(model.model.embed_tokens.parameters()).device
    context_limit = int(model.config.max_position_embeddings)
    source = "checkpoint"
    return (
        model,
        tokenizer,
        prefix_cache,
        input_device,
        timestep,
        context_limit,
        source,
    )


def encode(tokenizer, messages, limit, device):
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        mask = torch.ones_like(ids)
    else:
        text = "".join(
            f"{message['role']}：{message['content']}\n" for message in messages
        )
        encoded = tokenizer(text + "assistant：", return_tensors="pt")
        ids, mask = encoded.input_ids, encoded.attention_mask
    if ids.shape[1] > limit:
        ids, mask = ids[:, -limit:], mask[:, -limit:]
    return ids.to(device), mask.to(device)


def fresh_cache(prefix_cache):
    if prefix_cache is None:
        return None, 0
    cache = DynamicCache.from_legacy_cache(
        tuple((key.clone(), value.clone()) for key, value in prefix_cache)
    )
    return cache, cache.get_seq_length()


def generation_kwargs(args, model, tokenizer):
    config = model.generation_config
    pad_token_id = config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    eos_token_id = config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    output = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "repetition_penalty": getattr(config, "repetition_penalty", 1.0),
    }
    if args.greedy:
        output.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        output.update(
            do_sample=True,
            temperature=(
                args.temperature
                if args.temperature is not None
                else getattr(config, "temperature", 0.7)
            ),
            top_p=(
                args.top_p
                if args.top_p is not None
                else getattr(config, "top_p", 0.8)
            ),
            top_k=(
                args.top_k
                if args.top_k is not None
                else getattr(config, "top_k", 20)
            ),
        )
    return output


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens 必须大于 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    (
        model,
        tokenizer,
        prefix_cache,
        device,
        timestep,
        context_limit,
        neuron_source,
    ) = load_model(args)
    _, prefix_length = fresh_cache(prefix_cache)
    available_input = context_limit - prefix_length - args.max_new_tokens
    if available_input <= 0:
        raise ValueError(
            f"上下文空间不足：context={context_limit}, prefix={prefix_length}, "
            f"max_new_tokens={args.max_new_tokens}"
        )
    input_limit = (
        min(args.max_input_tokens, available_input)
        if args.max_input_tokens is not None
        else available_input
    )
    if input_limit <= 0:
        raise ValueError("--max-input-tokens 必须大于 0")

    messages = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt
        else []
    )
    print(f"已加载 Phase-SNN: {args.model_path} (T={timestep})")
    print(f"神经元参数来源: {neuron_source}")
    print(f"最大输入长度: {input_limit} token；最大生成长度: {args.max_new_tokens} token")
    print("输入 /clear 清空上下文；输入 exit 或 quit 退出。")

    while True:
        try:
            question = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if question == "/clear":
            messages = (
                messages[:1]
                if messages and messages[0]["role"] == "system"
                else []
            )
            print("上下文已清空。")
            continue
        if not question:
            continue

        pending = messages + [{"role": "user", "content": question}]
        ids, mask = encode(tokenizer, pending, input_limit, device)
        past, prefix_length = fresh_cache(prefix_cache)
        if prefix_length:
            prefix_mask = torch.ones(
                (mask.shape[0], prefix_length),
                dtype=mask.dtype,
                device=mask.device,
            )
            mask = torch.cat((prefix_mask, mask), dim=1)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=past,
                **generation_kwargs(args, model, tokenizer),
            )
        answer = tokenizer.decode(
            outputs[0, ids.shape[1] :], skip_special_tokens=True
        ).strip()
        print(f"助手：{answer}")
        messages = pending + [{"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
