'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-04 23:35:08
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-05-28 23:10:17
'''
import logging
import os
from contextlib import nullcontext
import pathlib

TRL_USE_RICH = os.getenv("TRL_USE_RICH", False)

from trl.scripts import ScriptArguments, TrlParser
from trl import (
    ModelConfig,
    SFTConfig,
    get_quantization_config,
    get_kbit_device_map,
)

if TRL_USE_RICH:
    # RichProgressCallback lives under trl.trainer.callbacks which transitively imports
    # judges → llm_blender → mergekit in newer TRL releases. Only pay that cost when
    # the user actually opts into the rich progress bar.
    from trl.scripts import init_zero_verbose
    from trl import RichProgressCallback  # noqa: F401  (used below)
    init_zero_verbose()
    FORMAT = "%(message)s"
    from rich.console import Console
    from rich.logging import RichHandler
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)
    
from datasets import load_dataset,Dataset

import torch

# --- Resume compatibility shim (added for cross-version checkpoint resume) -------------
# transformers' Trainer._load_rng_state() calls torch.load(rng_file, weights_only=True).
# torch >= 2.6's safe unpickler rejects numpy globals, and the original training box saved
# the RNG state under numpy 2.x ("numpy._core.*") while this env pins numpy 1.26.4
# ("numpy.core.*"). Both break the resume. These checkpoints are our own (pulled from our
# R2 bucket) and therefore trusted, so we (a) alias numpy._core -> numpy.core for unpickling
# and (b) force weights_only=False on torch.load. This makes resume robust across both the
# current checkpoint and any future ones this run writes.
import sys as _sys
import numpy as _np
if not hasattr(_np, "_core"):
    try:
        import numpy.core as _npcore
        _sys.modules["numpy._core"] = _npcore
        for _sub in ("multiarray", "umath", "numeric", "_multiarray_umath"):
            try:
                _sys.modules[f"numpy._core.{_sub}"] = __import__(f"numpy.core.{_sub}", fromlist=[_sub])
            except Exception:
                pass
    except Exception:
        pass
_orig_torch_load = torch.load
def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # trusted local/R2 checkpoints; numpy in rng_state
    return _orig_torch_load(*args, **kwargs)
torch.load = _trusted_torch_load
# --------------------------------------------------------------------------------------

from tqdm.rich import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Trainer,
)

import json
import re

from rich import print,console
from jarvisvla.inference import action_tokens
from jarvisvla.train.utils_train import (
    MoreConfig,
    assert_freeze_patterns_match,
    disable_thinking_mode,
    print_trainable_parameters,
    resize_aux_heads,
    seed_everything,
)
from jarvisvla import QWEN_SPECIAL_TOKENS
from jarvisvla.train.data_collator import make_collator
from jarvisvla.train.r2_callback import R2UploadCallback


class DifferentialLRTrainer(Trainer):
    """Trainer with a higher LR for token-embedding and lm_head matrices.

    Phase-1 SFT adds 73 randomly-initialized action-token rows to embed_tokens + lm_head.
    A global LR of ~3e-6 is too low to pull those random rows out of the noise floor in
    a few thousand steps; the pretrained rows in the same matrices tolerate a higher LR
    fine over short SFT, so we elevate the whole matrix.

    Override via env var:
        EMBED_LR  — LR for embed_tokens + lm_head (default 30× args.learning_rate)
    """

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        from torch.optim import AdamW

        args = self.args
        embed_lr = float(os.environ["EMBED_LR"]) if os.environ.get("EMBED_LR") else 30.0 * args.learning_rate

        try:
            decay_param_names = set(self.get_decay_parameter_names(self.model))
        except AttributeError:
            decay_param_names = {n for n, _ in self.model.named_parameters() if "bias" not in n.lower() and "norm" not in n.lower()}

        decay_params, no_decay_params, embed_params, embed_param_names = [], [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "embed_tokens" in name or name == "lm_head.weight" or name.endswith(".lm_head.weight"):
                embed_params.append(p)
                embed_param_names.append(name)
            elif name in decay_param_names:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        groups = []
        if decay_params:
            groups.append({"params": decay_params, "lr": args.learning_rate, "weight_decay": args.weight_decay})
        if no_decay_params:
            groups.append({"params": no_decay_params, "lr": args.learning_rate, "weight_decay": 0.0})
        if embed_params:
            groups.append({"params": embed_params, "lr": embed_lr, "weight_decay": 0.0})

        self.optimizer = AdamW(
            groups,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=getattr(args, "adam_epsilon", 1e-8),
        )
        if args.local_rank in (0, -1):
            print(
                f"[diff-lr] base_lr={args.learning_rate} embed_lr={embed_lr} "
                f"groups: decay={len(decay_params)} no_decay={len(no_decay_params)} embed={len(embed_params)}",
                flush=True,
            )
            for n in embed_param_names:
                print(f"[diff-lr]   embed group: {n}", flush=True)
        return self.optimizer


tqdm.pandas()    

if __name__ == "__main__":
    
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, MoreConfig))
    sft_script_args, training_args, model_config, more_cfg = parser.parse_args_and_config()

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    seed_everything(training_args.seed)

    ################
    # Model, Tokenizer & Processor
    ################
    
    backbone = more_cfg.backbone

    ### discard: if no chat_template is defined in tokenizer_config.json, use the default one
    DEFAULT_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['role'] + ':\n\n'+ message['content'] + '\n' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"""
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # Default to flash_attention_2 for production but respect explicit overrides (e.g.
    # --attn_implementation sdpa when flash-attn isn't installed yet).
    if not model_config.attn_implementation:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load processor + model + register all special tokens. Resizing of embeddings
    # (and any auxiliary heads like MTP) happens once after the full token set is added.
    if backbone == "qwen2_vl":
        processor_config = dict(do_rescale=False, patch_size=14, vision_feature_select_strategy="default")
        processor = Qwen2VLProcessor.from_pretrained(model_config.model_name_or_path, **processor_config)
        with open(QWEN_SPECIAL_TOKENS, "r") as file:
            extra_specials = json.load(file)
        n_added = processor.tokenizer.add_special_tokens({"additional_special_tokens": extra_specials})
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif backbone == "qwen3_5":
        # Liger-Kernel fused-Triton patches (RMSNorm + SwiGLU + fused linear+CE).
        # Default ON; disable with LIGER=0 if any kernel mismatches the backbone.
        # Must be applied BEFORE the model is loaded so .from_pretrained picks up the
        # patched module classes.
        if os.environ.get("LIGER", "1") != "0":
            try:
                from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
                apply_liger_kernel_to_qwen3_5()
                if training_args.local_rank in (0, -1):
                    print("[liger] applied fused kernels to qwen3_5 (fused_linear_ce, rms_norm, swiglu)", flush=True)
            except ImportError:
                if training_args.local_rank in (0, -1):
                    print("[liger] not installed; skipping (pip install liger-kernel)", flush=True)
        processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)
        # Schema action tokens + the existing point/visual/think/grounding/etc. specials.
        schema = action_tokens.get_schema("qwen3_5")
        with open(QWEN_SPECIAL_TOKENS, "r") as file:
            extra_specials = json.load(file)
        all_specials = list(dict.fromkeys(schema.all_special_strings + extra_specials))
        n_added = processor.tokenizer.add_special_tokens({"additional_special_tokens": all_specials})
        model = AutoModelForImageTextToText.from_pretrained(model_config.model_name_or_path, **model_kwargs)
        # CoT during every env step exceeds the latency budget. Disable at training time so
        # the saved config doesn't accidentally enable it for vLLM serving downstream.
        disable_thinking_mode(model)
    else:
        raise ValueError(f"unsupported backbone {backbone!r}; known: qwen2_vl, qwen3_5")

    # CRITICAL: resize embeddings + MTP heads after add_special_tokens. Skipping this is
    # the silent-OOB bug all three frontier reviewers flagged.
    if n_added > 0:
        new_vocab_size = len(processor.tokenizer)
        print(f"[train] added {n_added} special tokens; resizing embeddings -> {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size)
        resize_aux_heads(model, new_vocab_size)
        # If lm_head was tied to input embeddings, retie after resize.
        if getattr(model.config, "tie_word_embeddings", False):
            model.tie_weights()

    # Validate action-token resolution against the live tokenizer. This catches naming
    # mismatches between the schema and what we actually registered.
    action_tokens.build_id_maps(action_tokens.get_schema(backbone), processor.tokenizer)

    if not processor.tokenizer.chat_template:
        raise ValueError("No chat_template found in the tokenizer_config.json, please set the chat_template in the tokenizer_config.json.")

    processor.tokenizer.padding_side = "right"
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Freeze patterns are backbone-specific. Qwen3.5-9B's parameter names need verification
    # in Week 2 — if these regexes match 0 parameters, the assertion below fails loudly.
    _FREEZE_PATTERNS = {
        "qwen2_vl": {
            "visual_encoder": [r"visual\.blocks.*", r"visual\.patch_embed.*"],
            "visual_adapter": [r"visual\.merger.*"],
            "language_backbone": [r"model\.embed_tokens.*", r"model\.layers.*"],
            "lm_head": [r"model\.norm.*", r"lm_head.*"],
        },
        "qwen3_5": {
            # Verified via tests/smoke_qwen3_5.py against /ephemeral/models/Qwen3.5-9B.
            # Qwen3.5-9B prefixes the multimodal submodules under `model.` (the visual
            # tower lives at model.visual.*, the LM at model.language_model.*).
            "visual_encoder": [r"model\.visual\.blocks.*", r"model\.visual\.patch_embed.*"],
            "visual_adapter": [r"model\.visual\.merger.*"],
            "language_backbone": [r"model\.language_model\.embed_tokens.*", r"model\.language_model\.layers.*"],
            "lm_head": [r"model\.language_model\.norm.*", r"lm_head.*"],
        },
    }
    fix_refexs: list[str] = []
    if getattr(more_cfg, "fix_visual_encoder", False):
        fix_refexs.extend(_FREEZE_PATTERNS[backbone]["visual_encoder"])
    if getattr(more_cfg, "fix_visual_adapter", False):
        fix_refexs.extend(_FREEZE_PATTERNS[backbone]["visual_adapter"])
    if getattr(more_cfg, "fix_language_backbone", False):
        fix_refexs.extend(_FREEZE_PATTERNS[backbone]["language_backbone"])
    if getattr(more_cfg, "fix_lm_head", False):
        fix_refexs.extend(_FREEZE_PATTERNS[backbone]["lm_head"])

    if fix_refexs:
        assert_freeze_patterns_match(model, fix_refexs)
    for name, param in model.named_parameters():
        if any(re.match(pattern, name) for pattern in fix_refexs):
            param.requires_grad = False
    
    
    ##################
    # DataCollator
    ##################

    # 找到image_fold
    image_fold = pathlib.Path(sft_script_args.dataset_name).parent
    image_fold = image_fold.parent if image_fold.name=="output" else image_fold
    data_collator = make_collator(more_cfg.collator_type,
                                  processor=processor,
                                  backbone=backbone,
                                  image_folder=image_fold,
                                  max_seq_length = training_args.max_seq_length,
                                  min_pixels = more_cfg.min_pixels,
                                  max_pixels = more_cfg.max_pixels,
                                  )
    
    ################
    # Dataset
    ################
    
    raw_datasets = load_dataset(sft_script_args.dataset_name)
    
    train_dataset = raw_datasets['train']
    train_dataset_len = train_dataset.num_rows
    train_dataset_len = int(more_cfg.dataset_p*train_dataset_len)
    train_dataset = train_dataset.shuffle(training_args.seed)
    if train_dataset_len < 0:
        select_ids = range(train_dataset.num_rows + train_dataset_len,train_dataset.num_rows)
    else:
        select_ids = range(train_dataset_len)
    train_dataset = train_dataset.select(select_ids)
    # HF datasets normalizes "valid" → "validation" when loading from a local parquet dir,
    # but the original HF dataset uses "valid". Accept either.
    eval_split_name = "valid" if "valid" in raw_datasets else "validation"
    eval_dataset = raw_datasets[eval_split_name]
    
    if training_args.local_rank in { 0 ,-1 }:
        print(train_dataset_len,more_cfg.dataset_p,int(more_cfg.dataset_p*train_dataset_len))
    
    ################
    # Optional rich context managers
    ###############
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    
    output_dir = pathlib.Path(training_args.output_dir)
    if output_dir.exists() and list(output_dir.glob("checkpoint-*")):
        training_args.resume_from_checkpoint = True

    # Disabled during training; restored before the final save so vLLM serves with KV cache.
    model.config.use_cache = False

    training_args.dataset_text_field = "text"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
        
    # HF transformers 5.x renamed `tokenizer` → `processing_class`. Pick whichever the
    # installed version accepts so the script works against both.
    # Callbacks: rich progress (opt-in via TRL_USE_RICH) and R2 upload (opt-in via env).
    callbacks = []
    if TRL_USE_RICH:
        callbacks.append(RichProgressCallback)
    r2_callback = R2UploadCallback.maybe_create(training_args.output_dir)
    if r2_callback is not None:
        callbacks.append(r2_callback)

    import inspect as _inspect
    _trainer_kwargs = dict(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=None,
        compute_metrics=None,
        callbacks=callbacks or None,
        preprocess_logits_for_metrics=None,
    )
    _sig = _inspect.signature(Trainer.__init__).parameters
    if "processing_class" in _sig:
        _trainer_kwargs["processing_class"] = processor.tokenizer
    else:
        _trainer_kwargs["tokenizer"] = processor.tokenizer
    trainer = DifferentialLRTrainer(**_trainer_kwargs)
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print_trainable_parameters(trainer.model,trainer.optimizer,f"logs/model_structure.json")

    if training_args.do_train:
        if output_dir.exists() and list(output_dir.glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    elif not training_args.do_train and training_args.do_eval:
        trainer.evaluate()

    if training_args.save_strategy != "no":
        model.config.use_cache = True
        trainer.save_model(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        # Upload the end-of-training save (model + processor at output_dir root, skipping
        # checkpoint-N subdirs which were already uploaded via on_save). Then block until
        # all in-flight uploads finish before process exit.
        if r2_callback is not None and training_args.local_rank in (0, -1):
            r2_callback.upload_path(training_args.output_dir, remote_subprefix="final")
            r2_callback.wait_all()

