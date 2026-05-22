"""Action ↔ token mapping for the Minecraft VLA.

Encoding flow (training data prep):
    minerl_action → group_action (12-vector) → token string → ... (consumed by HF tokenizer)

Decoding flow (inference):
    token_ids (from LLM) → group_actions → decimal_actions → action dicts (handed to env)

The token-string ↔ token-ID resolution lives in jarvisvla.inference.action_tokens and is
performed once at startup against a specific HF tokenizer. This module is tokenizer-agnostic
at the encode/decode layer — it operates on the precomputed maps in ActionIdMaps.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Union

import numpy as np
import torch
from rich import console

from minestudio.simulator.entry import CameraConfig
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.utils.vpt_lib.actions import ActionTransformer, Buttons

from jarvisvla.inference.action_tokens import (
    ActionIdMaps,
    ActionSchema,
    build_id_maps,
    get_schema,
)


class ActionTokenizer(ABC):
    """Base class for action tokenizers. Holds the schema/ID maps and the camera
    quantization machinery; subclasses implement encode/decode."""

    movements = ("forward", "back", "left", "right", "sprint", "sneak")
    operations = ("use", "drop", "attack", "jump")

    def __init__(
        self,
        maps: ActionIdMaps,
        camera_quantization_scheme: str = "mu_law",
        camera_mu: int = 20,
        camera_binsize: int = 1,
        camera_maxval: int = 10,
    ):
        self.maps = maps

        camera_config = CameraConfig(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            camera_quantization_scheme=camera_quantization_scheme,
            camera_mu=camera_mu,
        )
        self.n_camera_bins = camera_config.n_camera_bins
        self.null_action = {
            "forward": False, "back": False, "left": False, "right": False,
            "sprint": False, "sneak": False,
            "hotbar.1": False, "hotbar.2": False, "hotbar.3": False, "hotbar.4": False,
            "hotbar.5": False, "hotbar.6": False, "hotbar.7": False, "hotbar.8": False, "hotbar.9": False,
            "use": False, "drop": False, "attack": False, "jump": False,
            "inventory": False,
            "camera": (0.0, 0.0),
        }
        self.action_transformer = ActionTransformer(**camera_config.action_transformer_kwargs)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=camera_config.n_camera_bins)

    # ---- Schema / ID accessors (cheap pass-through to self.maps) ----

    @property
    def schema(self) -> ActionSchema:
        return self.maps.schema

    @property
    def bases(self) -> tuple[int, ...]:
        return self.schema.bases

    @property
    def tokenizer_type(self) -> str:
        """Back-compat alias used by external code that expects the old field name."""
        return self.schema.backbone

    @property
    def act_beg_id(self) -> int:
        return self.maps.act_beg_id

    @property
    def act_end_id(self) -> int:
        return self.maps.act_end_id

    @property
    def act_beg_token(self) -> str:
        return self.schema.act_start

    @property
    def act_end_token(self) -> str:
        return self.schema.act_end

    # ---- Subclass interface ----

    @abstractmethod
    def encode(self, actions: Dict) -> Union[torch.Tensor, list, str]:
        ...

    @abstractmethod
    def decode(self, tokens: Union[torch.Tensor, list]) -> List[OrderedDict]:
        ...


class OneActionTokenizer(ActionTokenizer):
    """Single-action tokenizer for the 12-group action vocabulary."""

    BUTTONS_GROUPS = [
        "hotbar", "fore or back", "left or right", "sprint or sneak", "use",
        "drop", "attack", "jump", "camera"
    ]

    def __init__(
        self,
        maps: ActionIdMaps,
        camera_quantization_scheme: str = "mu_law",
        camera_mu: int = 20,
        camera_binsize: int = 1,
    ):
        super().__init__(
            maps=maps,
            camera_quantization_scheme=camera_quantization_scheme,
            camera_mu=camera_mu,
            camera_binsize=camera_binsize,
        )
        console.Console().log(f"backbone: {self.schema.backbone}")
        console.Console().log(
            f"bases: {self.bases}, camera_mu: {camera_mu}, "
            f"n_camera_bins: {self.n_camera_bins}, camera_binsize: {camera_binsize}"
        )
        # NULL_ACTION: null buttons + middle camera bins encoded as a single decimal pair.
        self.NULL_ACTION = [0, (self.bases[-2] // 2) * self.bases[-2] + (self.bases[-1] // 2)]

    @classmethod
    def from_tokenizer(cls, backbone: str, tokenizer: Any, **kwargs: Any) -> "OneActionTokenizer":
        """Convenience constructor: resolve a backbone's schema against an HF tokenizer
        and build the ID maps. Call this after add_special_tokens/resize_token_embeddings."""
        schema = get_schema(backbone)
        maps = build_id_maps(schema, tokenizer)
        return cls(maps=maps, **kwargs)

    # ---- Decode: tokens → action dicts ----

    def decode(self, tokens: Union[torch.Tensor, List]) -> List[OrderedDict]:
        group_actions = self.token_2_group_action(tokens)
        actions = [self.group_action_2_decimal_action(ga) for ga in group_actions]
        out: list[OrderedDict] = []
        for buttons, camera in actions:
            out.append(OrderedDict(buttons=np.array([buttons])[0], camera=np.array([camera])[0]))
        return out

    def token_2_group_action(self, tokens: Union[torch.Tensor, list]) -> list[list[int]]:
        """Split a token-ID stream on <act_start>/<act_end> tags, decode each segment to a
        12-vector group_action. The camera flag at group -4 is reconstructed from the
        camera bins (not stored in the stream)."""
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim == 2:
                tokens = tokens.squeeze()
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            raise TypeError(f"tokens must be tensor or list, got {type(tokens).__name__}")

        camera_null = [self.bases[-1] // 2, self.bases[-2] // 2]
        action_base = [0] * len(self.bases)
        action_base[-2:] = camera_null

        actions: list[list[int]] = []
        start_idx = 0
        while start_idx < len(tokens):
            try:
                beg = tokens.index(self.act_beg_id, start_idx)
                end = tokens.index(self.act_end_id, beg + 1)
            except ValueError:
                break
            control_tokens = tokens[beg + 1:end]
            action = copy.copy(action_base)
            for token in control_tokens:
                place_value = self.maps.token_to_action.get(token)
                if place_value is None:
                    # Unknown token inside an action segment — silently ignored at inference
                    # for robustness; training/validation should set strict=True externally.
                    continue
                place, num = place_value
                action[place] = num
            # Reconstruct camera flag (group -4) from camera bins.
            if action[-2:] != camera_null:
                action[-4] = 1
            actions.append(copy.copy(action))
            start_idx = end + 1

        if not actions:
            actions.append(action_base)
        return actions

    def group_action_2_decimal_action(self, inputs: list[int]) -> tuple[int, int]:
        """Inverse of decimal_action_2_group_action."""
        if len(inputs) != len(self.bases):
            raise ValueError(
                f"input length {len(inputs)} != expected {len(self.bases)}"
            )
        decimal_results = [0, 0]
        mid = len(inputs) - 3  # boundary between buttons and camera
        for i, digit in enumerate(inputs):
            if digit >= self.bases[i]:
                raise ValueError(f"digit at position {i}={digit} exceeds base {self.bases[i] - 1}")
            if i < mid:
                decimal_results[0] = decimal_results[0] * self.bases[i] + digit
            elif i == mid and digit:
                decimal_results[0] = 8640  # inventory sentinel
            else:
                decimal_results[1] = decimal_results[1] * self.bases[i] + digit
        return tuple(decimal_results)

    # ---- Encode: action dicts → tokens (string form, fed to HF tokenizer) ----

    def encode(self, trajectory: dict) -> list[dict]:
        """Encode a trajectory (dict of action arrays) to a list of token-string records.
        Used for dataset prep, NOT inference. Inference uses encode_action directly."""
        minerl_actions = trajectory["actions"]
        traj_len = len(minerl_actions["attack"])
        observations = trajectory.get("observations", [""] * traj_len)
        frame_ids = trajectory.get("frame_ids", range(0, traj_len))
        uuids = trajectory.get("uuids", [""] * traj_len)

        minerl_action_transformed = {
            key: np.array(val)
            for key, val in minerl_actions.items()
            if key in Buttons.ALL or key == "camera"
        }
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        actions = self.action_mapper.from_factored(minerl_action)

        encoded: list[dict] = []
        for idx in range(traj_len):
            action_pair = (actions["buttons"][idx][0], actions["camera"][idx][0])
            encoded.append({
                "action_token": self.encode_action(action_pair),
                "observations": [observations[idx]],
                "uuid": uuids[idx],
                "frames": (frame_ids[idx], 1, frame_ids[idx]),
            })
        return encoded

    def encode_action(self, action: tuple) -> str:
        """Encode a (buttons_decimal, camera_decimal) pair to a token-string segment."""
        assert len(action) == 2
        group_action = self.decimal_action_2_group_action(action)
        return self.group_action_2_token(group_action)

    def group_action_2_token(self, group_action: list[int]) -> str:
        """Render a 12-vector group_action as a token-string segment.

        Layout:
            <|act_start|>  [buttons_sparse...]  [inventory_if_set]  camera_a  camera_b  <|act_end|>

        Group -4 (camera flag) is reconstructed at decode time from the camera bins, so
        we do not emit it. Group -3 (inventory flag) is sparse: emit only when set, but
        explicit emission IS required or inventory actions silently no-op.
        """
        parts: list[str] = [self.schema.act_start]
        # Sparse buttons: groups 0..-5.
        for g, num in enumerate(group_action[:-4]):
            if num != 0:
                parts.append(self.schema.group_tokens[g][num])
        # Inventory (group -3): emit only if set.
        inv_idx = len(group_action) - 3
        if group_action[inv_idx] != 0:
            parts.append(self.schema.group_tokens[inv_idx][group_action[inv_idx]])
        # Camera bins (groups -2, -1): always emit.
        for g_off in (-2, -1):
            g = len(group_action) + g_off
            parts.append(self.schema.group_tokens[g][group_action[g]])
        parts.append(self.schema.act_end)
        return "".join(parts)

    def decimal_action_2_group_action(self, inputs: tuple) -> tuple[int, ...]:
        """Decompose (buttons_decimal, camera_decimal) into the 12-vector group_action."""
        decimals = list(inputs)
        result = [0] * len(self.bases)
        inventory_flag = False

        if decimals[0] == 8640:
            inventory_flag = True
            decimals[0] = 0
        else:
            for i in range(len(self.bases) - 4, -1, -1):
                result[i] = decimals[0] % self.bases[i]
                decimals[0] //= self.bases[i]

        result[-1] = decimals[1] % self.bases[-1]
        decimals[1] //= self.bases[-1]
        result[-2] = decimals[1] % self.bases[-2]
        decimals[1] //= self.bases[-2]

        if inventory_flag:
            result[-3] = 1
        if decimals != [0, 0]:
            raise ValueError(f"decimal action too large for base system: leftover {decimals}")
        return tuple(result)

    def null_token(self) -> str:
        return self.encode_action(self.NULL_ACTION)
