"""CPU contracts for BAGEL's request-major image-step packing.

These tests intentionally stop before the model forward. They make the
silent-failure boundary explicit: local token indexes from a second request
must never address the first request's BOI/latent/EOI block.
"""

from __future__ import annotations

import unittest

import torch

from mstar.model.bagel.batching import (
    cfg_branch_labels_for_step,
    collate_image_gen_inputs,
    combine_cfg_velocities,
    combine_pruned_cfg_velocities,
    shifted_timestep_for_step,
    split_image_gen_outputs,
)
from mstar.model.bagel.submodules import _store_projected_image_latents
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
    def test_projected_latents_are_stored_in_sequence_dtype(self):
        combined = torch.zeros((5, 2), dtype=torch.float32)
        indexes = torch.tensor([1, 2, 3])
        projected = torch.full((3, 2), 1.5, dtype=torch.bfloat16)

        actual = _store_projected_image_latents(
            combined, indexes, projected
        )

        self.assertIs(actual, combined)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(
            actual[indexes], projected.to(dtype=torch.float32)
        )

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


class HeterogeneousCFGTest(unittest.TestCase):
    def _combine(
        self, timestep, latent_lens, renorm_type="global", rows=slice(None)
    ):
        v_main = torch.tensor([
            [1.0, 2.0], [2.0, 1.0],
            [10.0, 1.0], [1.0, 10.0],
        ])[rows]
        v_text = torch.tensor([
            [0.5, 0.0], [0.0, 0.5],
            [20.0, 2.0], [2.0, 20.0],
        ])[rows]
        v_image = torch.tensor([
            [0.25, 0.25], [0.25, 0.25],
            [30.0, 3.0], [3.0, 30.0],
        ])[rows]
        return combine_cfg_velocities(
            v_main, v_text, v_image, torch.tensor(timestep), latent_lens,
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_interval=(0.4, 1.0),
            cfg_renorm_type=renorm_type,
            cfg_renorm_min=0.0,
        )

    def test_global_renorm_matches_independent_request_combinations(self):
        batched = self._combine([0.8, 0.8, 0.2, 0.2], [2, 2])
        first = self._combine([0.8, 0.8], [2], rows=slice(0, 2))
        second = self._combine([0.2, 0.2], [2], rows=slice(2, 4))

        torch.testing.assert_close(batched, torch.cat([first, second]))

    def test_request_outside_cfg_interval_reduces_to_main_velocity(self):
        combined = self._combine([0.8, 0.8, 0.2, 0.2], [2, 2])
        expected_main = torch.tensor([[10.0, 1.0], [1.0, 10.0]])

        torch.testing.assert_close(combined[2:], expected_main)

    def test_all_supported_renorm_modes_keep_shape_and_finiteness(self):
        for renorm_type in ("global", "channel", "text_channel"):
            with self.subTest(renorm_type=renorm_type):
                combined = self._combine(
                    [0.8, 0.8, 0.2, 0.2], [2, 2], renorm_type
                )
                self.assertEqual(combined.shape, (4, 2))
                self.assertTrue(torch.isfinite(combined).all())

    def test_single_request_cfg_matches_the_previous_scalar_formula(self):
        v_main = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        v_text = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        v_image = torch.tensor([[0.25, 0.25], [0.25, 0.25]])

        for timestep_value in (0.8, 0.2):
            in_interval = float(0.4 < timestep_value <= 1.0)
            text_scale = 4.0 * in_interval + 1.0 * (1 - in_interval)
            image_scale = 1.5 * in_interval + 1.0 * (1 - in_interval)
            for renorm_type in ("global", "channel", "text_channel"):
                with self.subTest(
                    timestep=timestep_value, renorm_type=renorm_type
                ):
                    text_guided = v_text + text_scale * (v_main - v_text)
                    if renorm_type == "text_channel":
                        scale = (
                            v_main.norm(dim=-1, keepdim=True)
                            / (text_guided.norm(dim=-1, keepdim=True) + 1e-8)
                        ).clamp(min=0.0, max=1.0)
                        text_guided = text_guided * scale
                        expected = (
                            v_image + image_scale * (text_guided - v_image)
                            if image_scale > 1.0 else text_guided
                        )
                    else:
                        combined = (
                            v_image + image_scale * (text_guided - v_image)
                            if image_scale > 1.0 else text_guided
                        )
                        norm = (
                            v_main.norm(dim=-1, keepdim=True)
                            / (combined.norm(dim=-1, keepdim=True) + 1e-8)
                            if renorm_type == "channel"
                            else v_main.norm() / (combined.norm() + 1e-8)
                        )
                        expected = combined * norm.clamp(min=0.0, max=1.0)

                    actual = combine_cfg_velocities(
                        v_main,
                        v_text,
                        v_image,
                        torch.full((2,), timestep_value),
                        [2],
                        cfg_text_scale=4.0,
                        cfg_img_scale=1.5,
                        cfg_interval=(0.4, 1.0),
                        cfg_renorm_type=renorm_type,
                        cfg_renorm_min=0.0,
                    )
                    torch.testing.assert_close(actual, expected)


class CFGBranchPruningTest(unittest.TestCase):
    _DEFAULTS = {
        "requires_cfg": True,
        "num_timesteps": 50,
        "timestep_shift": 3.0,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": (0.4, 1.0),
    }

    def _labels(self, step: int, policy: str, **overrides):
        kwargs = {**self._DEFAULTS, **overrides, "policy": policy}
        return cfg_branch_labels_for_step(step, **kwargs)

    def test_default_cfg_boundary_is_derived_as_steps_40_to_41(self):
        self.assertGreater(
            shifted_timestep_for_step(
                40, num_timesteps=50, timestep_shift=3.0
            ),
            0.4,
        )
        self.assertLessEqual(
            shifted_timestep_for_step(
                41, num_timesteps=50, timestep_shift=3.0
            ),
            0.4,
        )
        self.assertEqual(
            self._labels(40, "auto"), ("main", "cfg_text")
        )
        self.assertEqual(self._labels(41, "auto"), ("main",))

    def test_experiment_arms_have_the_intended_branch_counts(self):
        expected = {
            "reference": (3, 3),
            "drop_image": (2, 2),
            "late_main": (3, 1),
            "auto": (2, 1),
            "main_only": (1, 1),
        }
        for policy, counts in expected.items():
            with self.subTest(policy=policy):
                self.assertEqual(len(self._labels(40, policy)), counts[0])
                self.assertEqual(len(self._labels(41, policy)), counts[1])

    def test_image_guidance_is_not_pruned_when_its_scale_is_active(self):
        for step in (0, 48):
            with self.subTest(step=step):
                self.assertEqual(
                    self._labels(
                        step,
                        "auto",
                        cfg_img_scale=2.0,
                        cfg_interval=(0.0, 1.0),
                    ),
                    ("main", "cfg_text", "cfg_img"),
                )

    def test_cfg_off_always_uses_main_only(self):
        for policy in (
            "reference", "drop_image", "late_main", "auto", "main_only"
        ):
            self.assertEqual(
                self._labels(0, policy, requires_cfg=False), ("main",)
            )

    def test_all_three_rows_delegate_to_reference_formula_exactly(self):
        main = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        text = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        image = torch.tensor([[0.25, 0.25], [0.25, 0.25]])
        timestep = torch.full((2,), 0.8)
        kwargs = dict(
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_interval=(0.4, 1.0),
            cfg_renorm_type="global",
            cfg_renorm_min=0.0,
        )

        reference = combine_cfg_velocities(
            main, text, image, timestep, [2], **kwargs
        )
        pruned_helper = combine_pruned_cfg_velocities(
            {"main": main, "cfg_text": text, "cfg_img": image},
            timestep,
            [2],
            **kwargs,
        )

        self.assertTrue(torch.equal(reference, pruned_helper))

    def test_inactive_image_row_can_be_removed_from_default_text_to_image(self):
        main = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        text = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        image = torch.tensor([[30.0, 30.0], [30.0, 30.0]])
        timestep = torch.full((2,), 0.8)
        kwargs = dict(
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_interval=(0.4, 1.0),
            cfg_renorm_type="global",
            cfg_renorm_min=0.0,
        )

        reference = combine_cfg_velocities(
            main, text, image, timestep, [2], **kwargs
        )
        without_image = combine_pruned_cfg_velocities(
            {"main": main, "cfg_text": text}, timestep, [2], **kwargs
        )

        self.assertTrue(torch.equal(reference, without_image))

    def test_main_only_late_step_returns_main_velocity(self):
        main = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        actual = combine_pruned_cfg_velocities(
            {"main": main},
            torch.full((2,), 0.2),
            [2],
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_interval=(0.4, 1.0),
            cfg_renorm_type="global",
            cfg_renorm_min=0.0,
        )

        self.assertIs(actual, main)

    def test_compiled_caller_can_change_active_branch_mapping(self):
        """A three-row warm-up must not specialize later pruned calls."""
        main = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        text = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        image = torch.tensor([[0.25, 0.25], [0.25, 0.25]])
        early_timestep = torch.full((2,), 0.8)
        late_timestep = torch.full((2,), 0.2)
        kwargs = dict(
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_interval=(0.4, 1.0),
            cfg_renorm_type="global",
            cfg_renorm_min=0.0,
        )

        def caller(branches, timestep):
            return combine_pruned_cfg_velocities(
                branches, timestep, [2], **kwargs
            )

        compiled = torch.compile(caller, backend="eager", fullgraph=False)
        reference = compiled(
            {"main": main, "cfg_text": text, "cfg_img": image},
            early_timestep,
        )
        drop_image = compiled(
            {"main": main, "cfg_text": text}, early_timestep
        )
        main_only = compiled({"main": main}, late_timestep)

        torch.testing.assert_close(drop_image, reference)
        self.assertIs(main_only, main)


class ImageBatchOutputSplittingTest(unittest.TestCase):
    def test_outputs_return_to_their_request_ids(self):
        latents = torch.tensor([
            [1.0, 1.0], [1.0, 2.0],
            [2.0, 1.0], [2.0, 2.0],
        ])
        time_index = torch.tensor([4, 4, 21, 21])

        outputs = split_image_gen_outputs(
            ["request-a", "request-b"], latents, time_index, [2, 2]
        )

        self.assertEqual(set(outputs), {"request-a", "request-b"})
        torch.testing.assert_close(
            outputs["request-a"]["latents"][0], latents[:2]
        )
        torch.testing.assert_close(
            outputs["request-b"]["latents"][0], latents[2:]
        )
        torch.testing.assert_close(
            outputs["request-a"]["time_index"][0], torch.tensor([4, 4])
        )
        torch.testing.assert_close(
            outputs["request-b"]["time_index"][0], torch.tensor([21, 21])
        )

    def test_outputs_reject_inconsistent_boundaries(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            split_image_gen_outputs(
                ["request-a", "request-b"],
                torch.zeros(3, 2),
                torch.zeros(3),
                [2, 2],
            )


if __name__ == "__main__":
    unittest.main()
