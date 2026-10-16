"""CPU contracts for BAGEL's request-major image-step packing.

These tests intentionally stop before the model forward. They make the
silent-failure boundary explicit: local token indexes from a second request
must never address the first request's BOI/latent/EOI block.
"""

from __future__ import annotations

import unittest

import torch

from mstar.model.bagel.batching import collate_image_gen_inputs
from mstar.model.submodule_base import ARNodeInputs


def _row(*, value: float, time_index: int, latent_len: int = 3) -> ARNodeInputs:
    seq_len = latent_len + 2
    return ARNodeInputs(
        input_embeds=torch.full(
            (latent_len, 2), value, dtype=torch.float32
        ),
        input_seq_len=seq_len,
        tensor_inputs={
            "empty_combined_emb": torch.full(
                (seq_len, 2), value + 10, dtype=torch.float32
            ),
            "vae_position_ids": torch.arange(latent_len),
            "text_indexes": torch.tensor([0, seq_len - 1]),
            "vae_token_indexes": torch.arange(1, seq_len - 1),
            "text_mask": torch.tensor(
                [True, *([False] * latent_len), True]
            ),
            "time_index": torch.full((latent_len,), time_index),
        },
    )


class ImageBatchPackingTest(unittest.TestCase):
    def test_image_rows_offset_local_indexes_into_the_packed_sequence(self):
        first = _row(value=1, time_index=3)
        second = _row(value=2, time_index=20)
        first_text = first.tensor_inputs["text_indexes"].clone()
        second_text = second.tensor_inputs["text_indexes"].clone()

        packed = collate_image_gen_inputs([first, second])

        self.assertEqual(packed["seq_lens"], [5, 5])
        self.assertEqual(packed["latent_lens"], [3, 3])
        torch.testing.assert_close(
            packed["text_indexes"], torch.tensor([0, 4, 5, 9])
        )
        torch.testing.assert_close(
            packed["vae_token_indexes"], torch.tensor([1, 2, 3, 6, 7, 8])
        )
        torch.testing.assert_close(
            packed["time_index"], torch.tensor([3, 3, 3, 20, 20, 20])
        )
        torch.testing.assert_close(
            packed["input_embeds"],
            torch.tensor([
                [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                [2.0, 2.0], [2.0, 2.0], [2.0, 2.0],
            ]),
        )
        # Packing must not rewrite the request-owned source tensors.
        self.assertTrue(torch.equal(
            first.tensor_inputs["text_indexes"], first_text
        ))
        self.assertTrue(torch.equal(
            second.tensor_inputs["text_indexes"], second_text
        ))

    def test_reversing_rows_preserves_boundaries_and_reverses_payloads(self):
        first = _row(value=1, time_index=3)
        second = _row(value=2, time_index=20)

        packed = collate_image_gen_inputs([second, first])

        torch.testing.assert_close(
            packed["time_index"], torch.tensor([20, 20, 20, 3, 3, 3])
        )
        self.assertTrue(torch.all(packed["input_embeds"][:3] == 2))
        self.assertTrue(torch.all(packed["input_embeds"][3:] == 1))
        # Index ownership follows the new request order, not the values stored
        # in either row.
        torch.testing.assert_close(
            packed["vae_token_indexes"], torch.tensor([1, 2, 3, 6, 7, 8])
        )

    def test_image_packing_rejects_a_sequence_boundary_mismatch(self):
        row = _row(value=1, time_index=3)
        row.input_seq_len += 1

        with self.assertRaisesRegex(ValueError, "declares sequence length"):
            collate_image_gen_inputs([row])

    def test_image_packing_rejects_an_out_of_range_local_index(self):
        row = _row(value=1, time_index=3)
        row.tensor_inputs["vae_token_indexes"][-1] = row.input_seq_len

        with self.assertRaisesRegex(
            ValueError, "out-of-range vae_token_indexes"
        ):
            collate_image_gen_inputs([row])

    def test_image_packing_rejects_missing_required_state(self):
        row = _row(value=1, time_index=3)
        row.tensor_inputs.pop("time_index")

        with self.assertRaisesRegex(ValueError, "missing inputs.*time_index"):
            collate_image_gen_inputs([row])


if __name__ == "__main__":
    unittest.main()
