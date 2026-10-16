"""Pure tensor packing and CFG helpers for BAGEL model batches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.model.submodule_base import ARNodeInputs, StackingMethod


CFG_BRANCH_POLICIES = frozenset({
    "reference",
    "drop_image",
    "late_main",
    "auto",
    "main_only",
})
CFG_BRANCH_ORDER = ("main", "cfg_text", "cfg_img")


def shifted_timestep_for_step(
    step_index: int,
    *,
    num_timesteps: int,
    timestep_shift: float,
) -> float:
    """Return the shifted BAGEL timestep used by one Euler iteration."""
    if num_timesteps < 2:
        raise ValueError("BAGEL num_timesteps must be at least 2")
    if not 0 <= step_index < num_timesteps - 1:
        raise ValueError(
            f"BAGEL step index {step_index} is outside "
            f"[0, {num_timesteps - 2}]"
        )
    t_uniform = 1.0 - step_index / (num_timesteps - 1)
    return (
        timestep_shift * t_uniform
        / (1.0 + (timestep_shift - 1.0) * t_uniform)
    )


def cfg_branch_labels_for_step(
    step_index: int,
    *,
    requires_cfg: bool,
    policy: str,
    num_timesteps: int,
    timestep_shift: float,
    cfg_text_scale: float,
    cfg_img_scale: float,
    cfg_interval: tuple[float, float] | list[float],
) -> tuple[str, ...]:
    """Select the physical CFG rows for one BAGEL refinement step.

    ``reference`` is the unchanged three-row implementation. The other
    policies are experiment arms, not new defaults:

    * ``drop_image`` removes only a mathematically inactive image branch.
    * ``late_main`` uses the reference rows inside the CFG interval and only
      main outside it.
    * ``auto`` keeps exactly the branches required by the configured scales.
    * ``main_only`` is an intentionally unguided negative control.
    """
    if policy not in CFG_BRANCH_POLICIES:
        raise ValueError(
            f"Unsupported BAGEL cfg_branch_policy {policy!r}; expected one of "
            f"{sorted(CFG_BRANCH_POLICIES)}"
        )
    if not requires_cfg or policy == "main_only":
        return ("main",)
    if policy == "reference":
        return CFG_BRANCH_ORDER

    if len(cfg_interval) != 2:
        raise ValueError("BAGEL cfg_interval must contain exactly two values")
    timestep = shifted_timestep_for_step(
        step_index,
        num_timesteps=num_timesteps,
        timestep_shift=timestep_shift,
    )
    cfg_lo, cfg_hi = cfg_interval
    in_interval = cfg_lo < timestep <= cfg_hi

    if policy == "late_main":
        return CFG_BRANCH_ORDER if in_interval else ("main",)

    effective_text_scale = cfg_text_scale if in_interval else 1.0
    effective_img_scale = cfg_img_scale if in_interval else 1.0
    if policy == "drop_image":
        return (
            CFG_BRANCH_ORDER
            if effective_img_scale > 1.0
            else ("main", "cfg_text")
        )

    labels = ["main"]
    if effective_text_scale != 1.0:
        labels.append("cfg_text")
    # The reference formula selects the image-guided result only for > 1.
    if effective_img_scale > 1.0:
        labels.append("cfg_img")
    return tuple(labels)


def collate_image_gen_inputs(inputs: list[ARNodeInputs]) -> dict[str, Any]:
    """Pack independent BAGEL image rows without losing request boundaries.

    ``ARNodeInputs.collate(..., CAT)`` correctly concatenates the tensors, but
    image-generation token indexes are local to each request. A plain
    concatenation would leave every request pointing at the first request's
    BOI/latent/EOI block. Offset those indexes into the packed request-major
    sequence and retain the lengths needed to split model outputs back.

    This helper deliberately validates the current image-step layout. A new
    layout should fail here rather than run a plausible-looking forward with
    cross-request token mixing.
    """
    if not inputs:
        raise ValueError("BAGEL image generation requires at least one input")

    required = {
        "empty_combined_emb",
        "vae_position_ids",
        "text_indexes",
        "vae_token_indexes",
        "text_mask",
        "time_index",
    }
    seq_lens: list[int] = []
    latent_lens: list[int] = []
    text_indexes: list[torch.Tensor] = []
    vae_token_indexes: list[torch.Tensor] = []
    seq_offset = 0

    for row, inp in enumerate(inputs):
        missing = required.difference(inp.tensor_inputs)
        if missing:
            raise ValueError(
                f"BAGEL image-generation row {row} is missing inputs: "
                f"{sorted(missing)}"
            )
        if inp.input_embeds is None:
            raise ValueError(
                f"BAGEL image-generation row {row} has no latent input_embeds"
            )

        seq_len = int(inp.input_seq_len)
        latent_len = int(inp.input_embeds.shape[0])
        tensors = inp.tensor_inputs
        observed = {
            "empty_combined_emb": int(tensors["empty_combined_emb"].shape[0]),
            "text_mask": int(tensors["text_mask"].shape[0]),
        }
        if any(length != seq_len for length in observed.values()):
            raise ValueError(
                f"BAGEL image-generation row {row} declares sequence length "
                f"{seq_len}, but observed {observed}"
            )
        latent_observed = {
            "vae_position_ids": int(tensors["vae_position_ids"].shape[0]),
            "vae_token_indexes": int(tensors["vae_token_indexes"].numel()),
            "time_index": int(tensors["time_index"].shape[0]),
        }
        if any(length != latent_len for length in latent_observed.values()):
            raise ValueError(
                f"BAGEL image-generation row {row} has {latent_len} latent "
                f"rows, but observed {latent_observed}"
            )

        # Value checks are useful in CPU tests, but reading a CUDA scalar here
        # would synchronize the hot denoising path on every request and step.
        for name in ("text_indexes", "vae_token_indexes"):
            indexes = tensors[name]
            if indexes.device.type == "cpu" and indexes.numel() and (
                int(indexes.min().item()) < 0
                or int(indexes.max().item()) >= seq_len
            ):
                raise ValueError(
                    f"BAGEL image-generation row {row} has out-of-range "
                    f"{name} for sequence length {seq_len}"
                )

        seq_lens.append(seq_len)
        latent_lens.append(latent_len)
        text_indexes.append(tensors["text_indexes"] + seq_offset)
        vae_token_indexes.append(tensors["vae_token_indexes"] + seq_offset)
        seq_offset += seq_len

    result = ARNodeInputs.collate(inputs, stacking_method=StackingMethod.CAT)
    result["text_indexes"] = torch.cat(text_indexes)
    result["vae_token_indexes"] = torch.cat(vae_token_indexes)
    result["seq_lens"] = seq_lens
    result["latent_lens"] = latent_lens
    return result


def combine_cfg_velocities(
    v_main: torch.Tensor,
    v_cfg_text: torch.Tensor,
    v_cfg_img: torch.Tensor,
    timestep: torch.Tensor,
    latent_lens: list[int],
    *,
    cfg_text_scale: float,
    cfg_img_scale: float,
    cfg_interval: tuple[float, float] | list[float],
    cfg_renorm_type: str,
    cfg_renorm_min: float,
) -> torch.Tensor:
    """Apply BAGEL CFG without coupling independent request rows.

    Timesteps and effective CFG phases may differ between requests. Channel
    renormalization is naturally row-local; global renormalization must be
    computed independently over each request's latent block rather than over
    the entire temporary model batch.
    """
    if not (
        v_main.shape == v_cfg_text.shape == v_cfg_img.shape
        and v_main.shape[0] == timestep.numel() == sum(latent_lens)
    ):
        raise ValueError(
            "BAGEL CFG tensors, timesteps, and latent lengths do not describe "
            "the same packed rows"
        )
    if any(length <= 0 for length in latent_lens):
        raise ValueError("BAGEL CFG latent lengths must be positive")
    if cfg_renorm_type not in {"global", "channel", "text_channel"}:
        raise ValueError(
            f"Unsupported BAGEL CFG renormalization: {cfg_renorm_type!r}"
        )

    cfg_lo, cfg_hi = cfg_interval
    in_cfg_interval = ((timestep > cfg_lo) & (timestep <= cfg_hi)).to(
        dtype=v_main.dtype
    )
    scale_shape = (-1,) + (1,) * (v_main.ndim - 1)
    in_cfg_interval = in_cfg_interval.reshape(scale_shape)
    effective_text_scale = (
        cfg_text_scale * in_cfg_interval + (1 - in_cfg_interval)
    )
    effective_img_scale = (
        cfg_img_scale * in_cfg_interval + (1 - in_cfg_interval)
    )

    v_text_guided = v_cfg_text + effective_text_scale * (
        v_main - v_cfg_text
    )
    if cfg_renorm_type == "text_channel":
        norm_v = torch.norm(v_main, dim=-1, keepdim=True)
        norm_v_text = torch.norm(v_text_guided, dim=-1, keepdim=True)
        scale = (norm_v / (norm_v_text + 1e-8)).clamp(
            min=cfg_renorm_min, max=1.0
        )
        v_text_guided = v_text_guided * scale
        image_guided = v_cfg_img + effective_img_scale * (
            v_text_guided - v_cfg_img
        )
        return torch.where(
            effective_img_scale > 1.0, image_guided, v_text_guided
        )

    image_guided = v_cfg_img + effective_img_scale * (
        v_text_guided - v_cfg_img
    )
    v_combined = torch.where(
        effective_img_scale > 1.0, image_guided, v_text_guided
    )
    if cfg_renorm_type == "channel":
        renorm_scale = (
            v_main.norm(dim=-1, keepdim=True)
            / (v_combined.norm(dim=-1, keepdim=True) + 1e-8)
        ).clamp(min=cfg_renorm_min, max=1.0)
        return v_combined * renorm_scale

    main_chunks = v_main.split(latent_lens)
    combined_chunks = v_combined.split(latent_lens)
    outputs = []
    for main, combined in zip(main_chunks, combined_chunks, strict=True):
        renorm_scale = (
            main.norm() / (combined.norm() + 1e-8)
        ).clamp(min=cfg_renorm_min, max=1.0)
        outputs.append(combined * renorm_scale)
    return torch.cat(outputs)


def combine_pruned_cfg_velocities(
    branch_velocities: Mapping[str, torch.Tensor],
    timestep: torch.Tensor,
    latent_lens: list[int],
    *,
    cfg_text_scale: float,
    cfg_img_scale: float,
    cfg_interval: tuple[float, float] | list[float],
    cfg_renorm_type: str,
    cfg_renorm_min: float,
) -> torch.Tensor:
    """Apply BAGEL CFG when inactive physical branch rows were omitted.

    With all three branches this delegates to the untouched reference helper.
    With only ``main`` it returns that velocity directly. Intermediate label
    sets retain the reference arithmetic for every branch that was executed.
    """
    labels = tuple(
        label for label in CFG_BRANCH_ORDER if label in branch_velocities
    )
    if set(branch_velocities) != set(labels) or not labels or labels[0] != "main":
        raise ValueError(
            "BAGEL CFG branch velocities must contain main and only known labels"
        )
    v_main = branch_velocities["main"]
    if labels == CFG_BRANCH_ORDER:
        return combine_cfg_velocities(
            v_main,
            branch_velocities["cfg_text"],
            branch_velocities["cfg_img"],
            timestep,
            latent_lens,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_type=cfg_renorm_type,
            cfg_renorm_min=cfg_renorm_min,
        )
    if labels == ("main",):
        if v_main.shape[0] != timestep.numel() or v_main.shape[0] != sum(latent_lens):
            raise ValueError(
                "BAGEL main velocity, timesteps, and latent lengths do not "
                "describe the same packed rows"
            )
        return v_main
    if any(length <= 0 for length in latent_lens):
        raise ValueError("BAGEL CFG latent lengths must be positive")
    if cfg_renorm_type not in {"global", "channel", "text_channel"}:
        raise ValueError(
            f"Unsupported BAGEL CFG renormalization: {cfg_renorm_type!r}"
        )
    if len(cfg_interval) != 2:
        raise ValueError("BAGEL cfg_interval must contain exactly two values")
    for label, velocity in branch_velocities.items():
        if velocity.shape != v_main.shape:
            raise ValueError(
                f"BAGEL CFG branch {label!r} does not match main velocity shape"
            )
    if v_main.shape[0] != timestep.numel() or v_main.shape[0] != sum(latent_lens):
        raise ValueError(
            "BAGEL CFG velocities, timesteps, and latent lengths do not "
            "describe the same packed rows"
        )

    cfg_lo, cfg_hi = cfg_interval
    in_cfg_interval = ((timestep > cfg_lo) & (timestep <= cfg_hi)).to(
        dtype=v_main.dtype
    )
    scale_shape = (-1,) + (1,) * (v_main.ndim - 1)
    in_cfg_interval = in_cfg_interval.reshape(scale_shape)
    effective_text_scale = (
        cfg_text_scale * in_cfg_interval + (1 - in_cfg_interval)
    )
    effective_img_scale = (
        cfg_img_scale * in_cfg_interval + (1 - in_cfg_interval)
    )

    if "cfg_text" in branch_velocities:
        v_cfg_text = branch_velocities["cfg_text"]
        v_text_guided = v_cfg_text + effective_text_scale * (
            v_main - v_cfg_text
        )
    else:
        # The runtime selector guarantees this condition without reading a
        # CUDA scalar back to the CPU. Retain the stronger check in CPU tests.
        if timestep.device.type == "cpu" and torch.any(
            effective_text_scale != 1.0
        ):
            raise ValueError("BAGEL cfg_text was pruned while its scale is active")
        v_text_guided = v_main

    if cfg_renorm_type == "text_channel":
        norm_v = torch.norm(v_main, dim=-1, keepdim=True)
        norm_v_text = torch.norm(v_text_guided, dim=-1, keepdim=True)
        scale = (norm_v / (norm_v_text + 1e-8)).clamp(
            min=cfg_renorm_min, max=1.0
        )
        v_text_guided = v_text_guided * scale
        if "cfg_img" not in branch_velocities:
            return v_text_guided
        v_cfg_img = branch_velocities["cfg_img"]
        image_guided = v_cfg_img + effective_img_scale * (
            v_text_guided - v_cfg_img
        )
        return torch.where(
            effective_img_scale > 1.0, image_guided, v_text_guided
        )

    if "cfg_img" in branch_velocities:
        v_cfg_img = branch_velocities["cfg_img"]
        image_guided = v_cfg_img + effective_img_scale * (
            v_text_guided - v_cfg_img
        )
        v_combined = torch.where(
            effective_img_scale > 1.0, image_guided, v_text_guided
        )
    else:
        v_combined = v_text_guided

    if cfg_renorm_type == "channel":
        renorm_scale = (
            v_main.norm(dim=-1, keepdim=True)
            / (v_combined.norm(dim=-1, keepdim=True) + 1e-8)
        ).clamp(min=cfg_renorm_min, max=1.0)
        return v_combined * renorm_scale

    main_chunks = v_main.split(latent_lens)
    combined_chunks = v_combined.split(latent_lens)
    outputs = []
    for main, combined in zip(main_chunks, combined_chunks, strict=True):
        renorm_scale = (
            main.norm() / (combined.norm() + 1e-8)
        ).clamp(min=cfg_renorm_min, max=1.0)
        outputs.append(combined * renorm_scale)
    return torch.cat(outputs)


def split_image_gen_outputs(
    request_ids: list[Any],
    latents: torch.Tensor,
    time_index: torch.Tensor,
    latent_lens: list[int],
) -> dict[Any, dict[str, list[torch.Tensor]]]:
    """Restore packed latent rows to their request-addressed output shape."""
    if len(request_ids) != len(latent_lens):
        raise ValueError(
            "BAGEL image outputs need one latent length per request"
        )
    total = sum(latent_lens)
    if latents.shape[0] != total or time_index.shape[0] != total:
        raise ValueError(
            "BAGEL image output tensors do not match their request boundaries"
        )
    latent_rows = latents.split(latent_lens)
    time_rows = time_index.split(latent_lens)
    return {
        rid: {
            "latents": [latent],
            "time_index": [step],
        }
        for rid, latent, step in zip(
            request_ids, latent_rows, time_rows, strict=True
        )
    }
