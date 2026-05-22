"""Tokenizer-agnostic description of the JarvisVLA action vocabulary.

Action tokens are 12 groups of values bracketed by start/end tags. The full action
grammar is:

    <|act_start|>  [group_0_token]?  [group_1_token]?  ...  [group_-3_token]?  group_-2_token  group_-1_token  <|act_end|>

where the trailing two tokens (camera bins) are mandatory and the others are
sparse (emitted only when nonzero). Group -4 (camera flag) is reconstructed at
decode time from the camera bins and is therefore never emitted.

Backbones differ in which *strings* hold those slots:
  * Qwen2-VL repurposed the existing `<|reserved_special_token_N|>` slots so the
    original checkpoint trained these IDs directly.
  * Qwen3.5+ doesn't expose a comparable block, so we use canonical names like
    `<|act_hotbar_0|>` added via `tokenizer.add_special_tokens(...)`.

This module owns the per-backbone string lists. Resolving those strings to
integer IDs against a specific tokenizer is `build_id_maps()`; that gets called
once at process startup, after any `add_special_tokens` and `resize_token_embeddings`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Group sizes — shared across backbones.
# Layout: hotbar, fb, lr, sprint/sneak, use, drop, attack, jump,
#         camera_flag, inventory, camera_bin_a, camera_bin_b
BASES: tuple[int, ...] = (10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21)


@dataclass(frozen=True)
class ActionSchema:
    backbone: str
    bases: tuple[int, ...]
    group_tokens: tuple[tuple[str, ...], ...]  # group_tokens[g][v] = token string
    act_start: str
    act_end: str

    def __post_init__(self) -> None:
        if len(self.bases) != len(self.group_tokens):
            raise ValueError(
                f"bases ({len(self.bases)}) and group_tokens ({len(self.group_tokens)}) length mismatch"
            )
        for g, (base, tokens) in enumerate(zip(self.bases, self.group_tokens)):
            if base != len(tokens):
                raise ValueError(f"group {g}: base {base} != token count {len(tokens)}")

    @property
    def all_special_strings(self) -> list[str]:
        out = [self.act_start, self.act_end]
        for group in self.group_tokens:
            out.extend(group)
        return out


def _qwen2_vl_reserved_strings() -> tuple[tuple[str, ...], ...]:
    """The exact reserved-special-token names the original Qwen2-VL checkpoint
    was SFT'd against. Do not reorder or rename — existing checkpoints encode
    action semantics into these specific token IDs (151833..151907)."""
    return (
        tuple(f"<|reserved_special_token_{180 + i}|>" for i in range(10)),    # hotbar
        tuple(f"<|reserved_special_token_{190 + i}|>" for i in range(3)),     # fb
        tuple(f"<|reserved_special_token_{193 + i}|>" for i in range(3)),     # lr
        tuple(f"<|reserved_special_token_{196 + i}|>" for i in range(3)),     # sprint/sneak
        tuple(f"<|reserved_special_token_{199 + i}|>" for i in range(2)),     # use
        tuple(f"<|reserved_special_token_{201 + i}|>" for i in range(2)),     # drop
        tuple(f"<|reserved_special_token_{203 + i}|>" for i in range(2)),     # attack
        tuple(f"<|reserved_special_token_{205 + i}|>" for i in range(2)),     # jump
        tuple(f"<|reserved_special_token_{207 + i}|>" for i in range(2)),     # camera_flag
        tuple(f"<|reserved_special_token_{176 + i}|>" for i in range(2)),     # inventory
        tuple(f"<|reserved_special_token_{209 + i}|>" for i in range(21)),    # camera_bin_a
        tuple(f"<|reserved_special_token_{230 + i}|>" for i in range(21)),    # camera_bin_b
    )


def _canonical_action_strings() -> tuple[tuple[str, ...], ...]:
    """Canonical, backbone-agnostic names for new backbones (Qwen3.5+) that
    register the tokens fresh via add_special_tokens()."""
    return (
        tuple(f"<|act_hotbar_{i}|>" for i in range(10)),
        tuple(f"<|act_fb_{n}|>" for n in ("null", "forward", "back")),
        tuple(f"<|act_lr_{n}|>" for n in ("null", "left", "right")),
        tuple(f"<|act_ss_{n}|>" for n in ("null", "sprint", "sneak")),
        tuple(f"<|act_use_{i}|>" for i in range(2)),
        tuple(f"<|act_drop_{i}|>" for i in range(2)),
        tuple(f"<|act_attack_{i}|>" for i in range(2)),
        tuple(f"<|act_jump_{i}|>" for i in range(2)),
        tuple(f"<|act_camflag_{i}|>" for i in range(2)),
        tuple(f"<|act_inv_{i}|>" for i in range(2)),
        tuple(f"<|act_cam_a_{i}|>" for i in range(21)),
        tuple(f"<|act_cam_b_{i}|>" for i in range(21)),
    )


QWEN2_VL_SCHEMA = ActionSchema(
    backbone="qwen2_vl",
    bases=BASES,
    group_tokens=_qwen2_vl_reserved_strings(),
    act_start="<|reserved_special_token_178|>",
    act_end="<|reserved_special_token_179|>",
)

QWEN3_5_SCHEMA = ActionSchema(
    backbone="qwen3_5",
    bases=BASES,
    group_tokens=_canonical_action_strings(),
    act_start="<|act_start|>",
    act_end="<|act_end|>",
)

_SCHEMAS = {s.backbone: s for s in (QWEN2_VL_SCHEMA, QWEN3_5_SCHEMA)}


def get_schema(backbone: str) -> ActionSchema:
    try:
        return _SCHEMAS[backbone]
    except KeyError:
        raise ValueError(
            f"no action schema for backbone {backbone!r}; available: {list(_SCHEMAS)}"
        )


@dataclass(frozen=True)
class ActionIdMaps:
    schema: ActionSchema
    token_to_action: dict[int, tuple[int, int]]
    action_to_token: dict[tuple[int, int], int]
    act_beg_id: int
    act_end_id: int


def build_id_maps(schema: ActionSchema, tokenizer: Any) -> ActionIdMaps:
    """Resolve a schema's string tokens against a tokenizer to integer IDs.

    Call this once at startup, AFTER any add_special_tokens / resize_token_embeddings.
    Raises on unknown tokens or duplicate IDs — both indicate the tokenizer wasn't
    set up correctly for this backbone.
    """
    unk = getattr(tokenizer, "unk_token_id", None)
    token_to_action: dict[int, tuple[int, int]] = {}
    action_to_token: dict[tuple[int, int], int] = {}

    def _resolve(tok_str: str, role: str) -> int:
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if tid is None or (unk is not None and tid == unk):
            raise ValueError(
                f"{role} token {tok_str!r} not found in tokenizer "
                f"(backbone={schema.backbone}); call add_special_tokens before build_id_maps"
            )
        return tid

    for g, group in enumerate(schema.group_tokens):
        for v, tok_str in enumerate(group):
            tid = _resolve(tok_str, f"action[group={g},value={v}]")
            if tid in token_to_action:
                other_g, other_v = token_to_action[tid]
                raise ValueError(
                    f"duplicate token id {tid}: ({g},{v})={tok_str!r} collides with "
                    f"({other_g},{other_v})={schema.group_tokens[other_g][other_v]!r}"
                )
            token_to_action[tid] = (g, v)
            action_to_token[(g, v)] = tid

    return ActionIdMaps(
        schema=schema,
        token_to_action=token_to_action,
        action_to_token=action_to_token,
        act_beg_id=_resolve(schema.act_start, "act_start"),
        act_end_id=_resolve(schema.act_end, "act_end"),
    )


def synthetic_id_maps(schema: ActionSchema, id_base: int = 1_000_000) -> ActionIdMaps:
    """Build an ActionIdMaps without a real tokenizer — assigns synthetic IDs.

    Used by unit tests that exercise encode/decode logic without needing to
    download a HuggingFace tokenizer. Real production paths use build_id_maps().
    """
    token_to_action: dict[int, tuple[int, int]] = {}
    action_to_token: dict[tuple[int, int], int] = {}
    next_id = id_base
    for g, group in enumerate(schema.group_tokens):
        for v in range(len(group)):
            token_to_action[next_id] = (g, v)
            action_to_token[(g, v)] = next_id
            next_id += 1
    act_beg_id = next_id; next_id += 1
    act_end_id = next_id
    return ActionIdMaps(
        schema=schema,
        token_to_action=token_to_action,
        action_to_token=action_to_token,
        act_beg_id=act_beg_id,
        act_end_id=act_end_id,
    )


def dump_action_token_map(maps: ActionIdMaps) -> dict:
    """Serializable view of the maps + schema for `action_token_map.json` next to the checkpoint."""
    return {
        "backbone": maps.schema.backbone,
        "bases": list(maps.schema.bases),
        "group_tokens": [list(g) for g in maps.schema.group_tokens],
        "act_start": maps.schema.act_start,
        "act_end": maps.schema.act_end,
        # Stored as (group, value) -> id for human readability; reverse map is derived.
        "action_to_token": {f"{g},{v}": tid for (g, v), tid in maps.action_to_token.items()},
        "act_beg_id": maps.act_beg_id,
        "act_end_id": maps.act_end_id,
    }


def verify_against_tokenizer(maps: ActionIdMaps, tokenizer: Any) -> None:
    """At inference startup, re-resolve schema strings against the loaded tokenizer
    and confirm IDs match the saved map. Catches tokenizer/checkpoint drift loudly."""
    live = build_id_maps(maps.schema, tokenizer)
    if live.action_to_token != maps.action_to_token:
        diffs = []
        for k in set(maps.action_to_token) | set(live.action_to_token):
            saved = maps.action_to_token.get(k)
            now = live.action_to_token.get(k)
            if saved != now:
                diffs.append(f"  {k}: saved={saved}, live={now}")
        raise RuntimeError(
            "action token IDs in loaded tokenizer do not match the saved checkpoint map. "
            "This usually means the tokenizer was modified after training, or the wrong "
            "backbone schema was loaded.\n" + "\n".join(diffs[:20])
        )
    if (live.act_beg_id, live.act_end_id) != (maps.act_beg_id, maps.act_end_id):
        raise RuntimeError(
            f"act_start/act_end ID mismatch: saved=({maps.act_beg_id},{maps.act_end_id}), "
            f"live=({live.act_beg_id},{live.act_end_id})"
        )
