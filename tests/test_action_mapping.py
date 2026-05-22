"""Round-trip tests for action_mapping.

Run with `python -m tests.test_action_mapping` from the repo root, or `pytest tests/`.

Uses `synthetic_id_maps` so the test does not need a real HF tokenizer — exercises
the encode/decode contract directly against the schema definitions.
"""
from __future__ import annotations

import sys

from jarvisvla.inference.action_mapping import OneActionTokenizer
from jarvisvla.inference.action_tokens import QWEN2_VL_SCHEMA, synthetic_id_maps


def _make_tokenizer(schema=QWEN2_VL_SCHEMA) -> OneActionTokenizer:
    return OneActionTokenizer(maps=synthetic_id_maps(schema))


def _string_to_ids(tok: OneActionTokenizer, token_string: str) -> list[int]:
    """Parse a token-string segment back to IDs by splitting on '<|...|>' markers.

    Mirrors what an HF tokenizer would do at inference for special tokens — every
    name is registered as a special token so it tokenizes atomically.
    """
    str_to_id: dict[str, int] = {tok.schema.act_start: tok.act_beg_id,
                                 tok.schema.act_end: tok.act_end_id}
    for g, group in enumerate(tok.schema.group_tokens):
        for v, s in enumerate(group):
            str_to_id[s] = tok.maps.action_to_token[(g, v)]

    ids: list[int] = []
    remaining = token_string
    while remaining:
        if not remaining.startswith("<|"):
            raise ValueError(f"unexpected non-special content: {remaining!r}")
        end = remaining.index("|>") + 2
        chunk, remaining = remaining[:end], remaining[end:]
        ids.append(str_to_id[chunk])
    return ids


def _camera_null_bin(tok: OneActionTokenizer) -> int:
    return tok.bases[-1] // 2


def _expected_camera_flag(group_action: list[int], camera_null: int) -> int:
    return 1 if group_action[-2:] != [camera_null, camera_null] else 0


def _assert_roundtrip(name: str, group_action: list[int]) -> None:
    tok = _make_tokenizer()
    token_string = tok.group_action_2_token(group_action)
    ids = _string_to_ids(tok, token_string)
    decoded = tok.token_2_group_action(ids)
    assert len(decoded) == 1, f"{name}: expected 1 action, got {len(decoded)}"
    got = list(decoded[0])
    expected = list(group_action)
    expected[-4] = _expected_camera_flag(group_action, _camera_null_bin(tok))
    assert got == expected, (
        f"{name}: roundtrip mismatch\n  in:  {group_action}\n  exp: {expected}\n  got: {got}"
    )


def test_null_action():
    tok = _make_tokenizer()
    cn = _camera_null_bin(tok)
    _assert_roundtrip("null", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cn, cn])


def test_inventory_open():
    """Regression test: group -3 (inventory) must round-trip."""
    tok = _make_tokenizer()
    cn = _camera_null_bin(tok)
    _assert_roundtrip("inventory", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, cn, cn])


def test_jump_with_camera():
    tok = _make_tokenizer()
    cn = _camera_null_bin(tok)
    _assert_roundtrip("jump+camera", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, cn + 2, cn - 3])


def test_hotbar_and_attack():
    tok = _make_tokenizer()
    cn = _camera_null_bin(tok)
    _assert_roundtrip("hotbar+attack", [3, 0, 0, 0, 0, 0, 1, 0, 0, 0, cn, cn])


def test_inventory_with_camera():
    tok = _make_tokenizer()
    cn = _camera_null_bin(tok)
    _assert_roundtrip("inventory+camera", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, cn + 5, cn - 5])


def test_decimal_inventory_roundtrip():
    """Full decimal → group → tokens → group → decimal path for an inventory action.
    Before the group_action_2_token fix this round-tripped to (0, 220) — silent loss."""
    tok = _make_tokenizer()
    null_camera_decimal = (tok.bases[-2] // 2) * tok.bases[-2] + (tok.bases[-1] // 2)
    input_action = (8640, null_camera_decimal)
    group = tok.decimal_action_2_group_action(input_action)
    token_string = tok.group_action_2_token(list(group))
    ids = _string_to_ids(tok, token_string)
    decoded_group = tok.token_2_group_action(ids)[0]
    decimal_out = tok.group_action_2_decimal_action(list(decoded_group))
    assert decimal_out == input_action, (
        f"inventory decimal roundtrip failed: in={input_action}, out={decimal_out}"
    )


def test_schema_self_consistency():
    """Every action-token string in the Qwen2-VL schema is unique."""
    seen = set()
    for g, group in enumerate(QWEN2_VL_SCHEMA.group_tokens):
        for v, s in enumerate(group):
            assert s not in seen, f"duplicate string {s!r} at ({g},{v})"
            seen.add(s)
    assert QWEN2_VL_SCHEMA.act_start not in seen
    assert QWEN2_VL_SCHEMA.act_end not in seen


if __name__ == "__main__":
    tests = [
        test_null_action,
        test_inventory_open,
        test_jump_with_camera,
        test_hotbar_and_attack,
        test_inventory_with_camera,
        test_decimal_inventory_roundtrip,
        test_schema_self_consistency,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n      {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures}/{len(tests)} test(s) failed")
        sys.exit(1)
    print(f"\n{len(tests)}/{len(tests)} tests passed")
