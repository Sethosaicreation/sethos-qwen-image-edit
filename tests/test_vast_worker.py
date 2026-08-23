import os
import unittest

os.environ.setdefault("VAST_WORKER_TOKEN", "t" * 64)
os.environ.setdefault("AI_WORKER_SPOOL", "/tmp/sethos-qwen-vast-test-spool")

from vast_worker import VastJobRequest, photo_request


class VastWorkerContractTests(unittest.TestCase):
    def test_maps_generic_vast_payload(self):
        request = VastJobRequest(
            job_id="01234567-89ab-4def-8123-456789abcdef",
            provider="qwen-2511-edit",
            model="Qwen-Image-Edit-2511",
            task="image_edit",
            input={
                "reference_image_url": (
                    "https://sethosaicreation.fr/admin/api/influencer-studio.php"
                    "?action=input&id=inf_0123456789abcdef01234567&token=" + "a" * 64
                ),
                "prompt": "Change uniquement le décor.",
                "edit_mode": "background",
                "quality": "quality",
                "seed": -1,
            },
        )
        parsed = photo_request(request)
        self.assertEqual(parsed.edit_mode, "background")
        self.assertEqual(parsed.quality, "quality")
        self.assertEqual(parsed.steps, 40)

    def test_requires_reference(self):
        request = VastJobRequest(
            job_id="01234567-89ab-4def-8123-456789abcdef",
            provider="qwen-2511-edit",
            model="Qwen-Image-Edit-2511",
            task="image_edit",
            input={"prompt": "Change le décor."},
        )
        with self.assertRaises(ValueError):
            photo_request(request)


if __name__ == "__main__":
    unittest.main()
