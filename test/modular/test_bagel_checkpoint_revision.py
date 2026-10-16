import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mstar.model.bagel import bagel_model


class BagelCheckpointRevisionTest(unittest.TestCase):
    def test_revision_reaches_every_hub_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            config_path = tmp_path / "config.json"
            config_path.write_text("{}")
            calls = []

            def fake_hf_hub_download(**kwargs):
                calls.append(("file", kwargs))
                return str(config_path)

            def fake_snapshot_download(**kwargs):
                calls.append(("snapshot", kwargs))
                return str(tmp_path / "snapshot")

            class FakeTokenizer:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    calls.append(("tokenizer", {"args": args, **kwargs}))
                    return object()

            with (
                patch.object(bagel_model, "hf_hub_download", fake_hf_hub_download),
                patch.object(bagel_model, "snapshot_download", fake_snapshot_download),
                patch.object(bagel_model, "BagelTokenizer", FakeTokenizer),
                patch.object(bagel_model, "load_bagel_config", lambda _: object()),
                patch.object(
                    bagel_model,
                    "add_special_tokens",
                    lambda tokenizer: (tokenizer, {}, None),
                ),
            ):
                model = bagel_model.BagelModel(
                    model_path_hf="org/model",
                    cache_dir="/cache",
                    checkpoint_revision="model-revision",
                )
                model._download_hf()
                model.checkpoint_path()

            self.assertEqual(Path(model.repo), tmp_path / "snapshot")
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(call[1]["revision"] == "model-revision" for call in calls))


if __name__ == "__main__":
    unittest.main()
