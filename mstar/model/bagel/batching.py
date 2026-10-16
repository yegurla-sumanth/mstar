"""Pure tensor packing helpers for BAGEL model batches."""

from __future__ import annotations

from typing import Any

import torch

from mstar.model.submodule_base import ARNodeInputs, StackingMethod


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

        for name in ("text_indexes", "vae_token_indexes"):
            indexes = tensors[name]
            if indexes.numel() and (
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
