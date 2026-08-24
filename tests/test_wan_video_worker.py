import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("VAST_WORKER_TOKEN", "t" * 64)
os.environ.setdefault("AI_WORKER_SPOOL", "/tmp/sethos-wan22-test-spool")
os.environ.setdefault("WAN_MODEL_CACHE", "/tmp/sethos-wan22-test-model-cache")

from wan_video_worker import VastVideoJobRequest, effective_prompt, probe_video, video_parameters


class WanVideoWorkerContractTests(unittest.TestCase):
    def request(self, **changes):
        data = {
            "reference_image_url": (
                "https://sethosaicreation.fr/admin/api/influencer-studio.php"
                "?action=video-input&id=inf_0123456789abcdef01234567&token=" + "a" * 64
            ),
            "prompt": "She turns naturally toward the fixed camera.",
            "negative_prompt": "face drift, unstable anatomy",
            "duration": 15,
            "aspect_ratio": "9:16",
            "resolution": "720p",
            "seed": -1,
        }
        data.update(changes)
        return VastVideoJobRequest(
            job_id="01234567-89ab-4def-8123-456789abcdef",
            provider="wan-ti2v-5b",
            model="Wan2.2-TI2V-5B",
            task="video",
            input=data,
        )

    def test_maps_15_seconds_to_four_n_plus_one_frames(self):
        parsed = video_parameters(self.request(seed=42))
        self.assertEqual(parsed["frames"], 361)
        self.assertEqual(parsed["size"], "704*1280")
        self.assertEqual(parsed["seed"], 42)

    def test_requires_signed_reference(self):
        with self.assertRaises(ValueError):
            video_parameters(self.request(reference_image_url="https://example.org/reference.jpg"))

    def test_rejects_unsupported_duration(self):
        with self.assertRaises(ValueError):
            video_parameters(self.request(duration=12))

    def test_prompt_keeps_frame_zero_guard(self):
        prompt = effective_prompt("A natural glance.", "face drift")
        self.assertIn("immutable frame zero", prompt)
        self.assertIn("no lip-sync", prompt)

    @patch("wan_video_worker.subprocess.run")
    def test_probe_reads_vertical_mp4_contract(self, mocked_run):
        mocked_run.return_value.returncode = 0
        mocked_run.return_value.stdout = (
            '{"streams":[{"width":704,"height":1280,"avg_frame_rate":"24/1",'
            '"nb_read_packets":"361"}],"format":{"duration":"15.041"}}'
        )
        with tempfile.TemporaryDirectory() as temporary:
            metrics = probe_video(Path(temporary) / "result.mp4")
        self.assertEqual(metrics["frames"], 361)
        self.assertEqual(metrics["height"], 1280)


if __name__ == "__main__":
    unittest.main()
