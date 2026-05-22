'''
Backbone routing for inference. String-based dispatch from the checkpoint path
is brittle (the original `lower().replace('-','_')` does not normalize e.g.
`Qwen3.5-9B` to `qwen3_5`), so a `backbone` override is accepted and preferred.
'''


_KNOWN_BACKBONES = ("qwen2_vl", "qwen3_5")


def load_visual_model(checkpoint_path: str = "", backbone: str = "", **kwargs):
    """Resolve LLM and VLM backbone names from a checkpoint path or explicit override.

    Pass `backbone="qwen3_5"` (or whichever) for unambiguous routing. The path-substring
    fallback exists only for backward compat with the original Qwen2-VL checkpoints
    whose paths happen to contain "qwen2_vl".
    """
    if not checkpoint_path and not backbone:
        raise AssertionError("checkpoint_path or backbone is required")

    if backbone:
        if backbone not in _KNOWN_BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; known: {_KNOWN_BACKBONES}")
        return backbone, backbone

    normalized = checkpoint_path.lower().replace("-", "_").replace(".", "_")
    for bb in _KNOWN_BACKBONES:
        if bb in normalized:
            return bb, bb
    raise AssertionError(
        f"could not infer backbone from path {checkpoint_path!r}; "
        f"pass backbone=... explicitly. Known backbones: {_KNOWN_BACKBONES}"
    )
