import unittest

from inference import directed_prompt
from schema import CONTRACT_VERSION, parse_request


class DirectedPromptTests(unittest.TestCase):
    def request(self):
        return parse_request({
            "input": {
                "contract_version": CONTRACT_VERSION,
                "source_image": (
                    "https://sethosaicreation.fr/admin/api/influencer-studio.php"
                    "?action=input&id=inf_0123456789abcdef01234567&token=" + "a" * 64
                ),
                "style_image": (
                    "https://sethosaicreation.fr/admin/api/influencer-studio.php"
                    "?action=input&id=inf_0123456789abcdef01234567&token=" + "b" * 64
                ),
                "prompt": "Keep the same bedroom but use a new viewpoint near the window.",
                "edit_mode": "free",
                "fidelity": "identity",
                "aspect_ratio": "9:16",
                "quality": "quality",
                "seed": -1,
            }
        })

    def test_style_reference_does_not_lock_composition(self):
        prompt = directed_prompt(self.request())
        self.assertIn("reference board, not a composition template", prompt)
        self.assertIn("never inherit its camera position", prompt)

    def test_identity_mode_allows_requested_camera_change(self):
        prompt = directed_prompt(self.request())
        self.assertIn("camera angle", prompt)
        self.assertIn("may change only where the user instruction explicitly requests", prompt)
        self.assertNotIn("body proportions, pose, camera angle, expression", prompt)


if __name__ == "__main__":
    unittest.main()
