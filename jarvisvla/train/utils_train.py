'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-04 23:31:28
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-05 03:47:11
'''
import random
import re
import torch
import torch.nn as nn
import os
import numpy as np
import pathlib
from dataclasses import dataclass, field

IGNORE_TOKEN_ID = -100

@dataclass
class MoreConfig:
    dataset_p: float = field(default=1.0, metadata={"help": "Dataset parameter p"})
    collator_type: str = field(default="MultimodalChatDataCollatorforVLM", metadata={"help": "types of collator"})
    fix_visual_encoder: bool = field(default=False, metadata={"help": "fix visual encoder"})
    fix_visual_adapter: bool = field(default=False, metadata={"help": "fix visual adapter layer"})
    fix_language_backbone: bool = field(default=False, metadata={"help": "fix language backbone"})
    fix_lm_head: bool = field(default=False, metadata={"help": "fix language model head"})
    min_pixels: int = field(default=3136)
    max_pixels: int = field(default=2048*28*28)
    backbone: str = field(
        default="qwen2_vl",
        metadata={"help": "Backbone family. One of: qwen2_vl, qwen3_5"},
    )


def resize_aux_heads(model: torch.nn.Module, new_vocab_size: int) -> int:
    """Walk a model after resize_token_embeddings and resize any auxiliary output
    projections whose vocab dimension also needs to change (e.g. MTP/next-token heads).

    Returns the number of modules resized. Logs candidates loudly so misses are visible
    — for a new backbone, verify the named modules listed match the model's actual MTP
    head structure before trusting the resize.

    HF's standard model.resize_token_embeddings() only touches the main lm_head. MTP-style
    heads are separate nn.Linear modules with their own output projection to vocab; if
    you don't resize them and you added new tokens, inference produces silent OOB.
    """
    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lname = name.lower()
        # Skip the main lm_head (already handled by resize_token_embeddings).
        if lname.endswith("lm_head") and "mtp" not in lname:
            continue
        looks_like_aux_head = (
            ("mtp" in lname or "next_token" in lname or "draft" in lname)
            and (lname.endswith("head") or lname.endswith("proj") or "out_proj" in lname)
        )
        if looks_like_aux_head:
            candidates.append((name, module))

    if not candidates:
        print(
            "[resize_aux_heads] no MTP/aux head candidates found. Either the model has no "
            "such heads, or the naming convention doesn't match the heuristic "
            "(mtp/next_token/draft + head/proj/out_proj). If MTP is expected for this "
            "backbone, inspect model.named_modules() and resize manually."
        )
        return 0

    n_resized = 0
    for name, module in candidates:
        if module.out_features == new_vocab_size:
            continue  # already correct (likely tied with lm_head)
        old_size = module.out_features
        print(
            f"[resize_aux_heads] resizing {name}: ({module.in_features}, {old_size}) "
            f"-> ({module.in_features}, {new_vocab_size})"
        )
        new_linear = nn.Linear(
            module.in_features, new_vocab_size,
            bias=module.bias is not None,
            device=module.weight.device, dtype=module.weight.dtype,
        )
        with torch.no_grad():
            min_size = min(old_size, new_vocab_size)
            new_linear.weight[:min_size] = module.weight[:min_size]
            if module.bias is not None:
                new_linear.bias[:min_size] = module.bias[:min_size]
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, new_linear)
        n_resized += 1
    return n_resized


def disable_thinking_mode(model) -> bool:
    """Disable chain-of-thought / thinking mode on the loaded model in-place.

    Returns True if any thinking-mode setting was found and disabled. Quietly returns
    False if the model has no such config — that's the right behavior for backbones
    that never had thinking mode in the first place.

    JarvisVLA's executor must respond with raw action tokens; CoT inside every env step
    blows the 50ms latency budget.
    """
    disabled = False
    # Try known config locations across model families.
    for attr in ("enable_thinking", "thinking_mode", "use_thinking"):
        for cfg in (getattr(model, "generation_config", None), getattr(model, "config", None)):
            if cfg is not None and hasattr(cfg, attr):
                setattr(cfg, attr, False)
                disabled = True
                print(f"[disable_thinking_mode] set {cfg.__class__.__name__}.{attr} = False")
    return disabled


def assert_freeze_patterns_match(model, patterns: list[str]) -> None:
    """Assert each freeze regex matches at least one parameter name.

    Silent zero-match means full fine-tune of layers we thought we froze — caught
    too late if we wait until eval.
    """
    for pat in patterns:
        n_matched = sum(1 for n, _ in model.named_parameters() if re.match(pat, n))
        if n_matched == 0:
            raise ValueError(
                f"freeze pattern {pat!r} matched 0 parameters — model layout has shifted; "
                f"update train.py's freeze regexes for this backbone before continuing"
            )

def seed_everything(seed: int) -> None:
    """Set global random seed for reproducibility."""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_trainable_parameters(model:torch.nn.Module,optimizer:torch.optim.Optimizer=None,record_path = None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    model_shapes = []
    for name, parameter in model.named_parameters():
        if optimizer:
            optimizer_group_idx = None
            for idx,param_group in enumerate(optimizer.param_groups):
                for param in param_group["params"]:
                    if parameter is param:
                        optimizer_group_idx = idx
            model_shapes.append([parameter.requires_grad,name,parameter.shape,optimizer_group_idx])
        else:
            model_shapes.append([parameter.requires_grad,name,parameter.shape])
        all_param += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    import json
    if record_path:
        pathlib.Path(record_path).parent.mkdir(parents=True,exist_ok=True)
        with open(record_path,mode="w",encoding="UTF-8") as f:
            json.dump(model_shapes, f, indent=4)
        
        with open(record_path.replace(".json","-scratch.txt"),mode="w",encoding="UTF-8") as f:
            print(optimizer, file=f)
            print(model, file=f)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
