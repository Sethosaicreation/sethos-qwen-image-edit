import unittest

from schema import CONTRACT_VERSION, InputError, parse_request


class SchemaTests(unittest.TestCase):
    def valid_event(self):
        return {
            "input": {
                "contract_version": CONTRACT_VERSION,
                "source_image": "https://sethosaicreation.fr/admin/api/photo-editor-runpod.php?action=input&id=pe_0123456789abcdef01234567&slot=source&token=" + "a" * 64,
                "prompt": "Change uniquement la tenue.",
                "edit_mode": "outfit",
                "fidelity": "identity",
                "aspect_ratio": "source",
                "quality": "standard",
                "seed": -1,
            }
        }

    def test_valid_request(self):
        request = parse_request(self.valid_event())
        self.assertEqual(request.steps, 30)
        self.assertEqual(request.edit_mode, "outfit")

    def test_accepts_influencer_studio_source(self):
        event = self.valid_event()
        event["input"]["source_image"] = (
            "https://sethosaicreation.fr/admin/api/influencer-studio.php"
            "?action=input&id=inf_0123456789abcdef01234567&token=" + "c" * 64
        )
        self.assertIn("influencer-studio.php", parse_request(event).source_image_url)

    def test_rejects_influencer_url_as_style_reference(self):
        event = self.valid_event()
        event["input"]["style_image"] = (
            "https://sethosaicreation.fr/admin/api/influencer-studio.php"
            "?action=input&id=inf_0123456789abcdef01234567&token=" + "c" * 64
        )
        with self.assertRaises(InputError):
            parse_request(event)

    def test_rejects_untrusted_image_host(self):
        event = self.valid_event()
        event["input"]["source_image"] = "https://example.org/image.png"
        with self.assertRaises(InputError):
            parse_request(event)

    def test_rejects_contract_mismatch(self):
        event = self.valid_event()
        event["input"]["contract_version"] = "wrong"
        with self.assertRaises(InputError):
            parse_request(event)


if __name__ == "__main__":
    unittest.main()
