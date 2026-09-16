from __future__ import annotations

import unittest

from toolkit.model_crypto.envelope import parse_kek_material, unwrap_dek, wrap_dek


class ModelCryptoEnvelopeTest(unittest.TestCase):
    def test_raw_32_byte_kek_is_not_stripped(self):
        raw = b" " + (b"x" * 30) + b"\n"
        self.assertEqual(raw, parse_kek_material(raw))

    def test_envelope_associated_data_detects_tampering(self):
        kek = b"k" * 32
        dek = b"d" * 32
        envelope = wrap_dek(kek, dek, kek_id="test")
        self.assertEqual(dek, unwrap_dek(kek, envelope))

        tampered = dict(envelope)
        tampered["aad"] = b"wrong-context"
        with self.assertRaisesRegex(ValueError, "KEK 不匹配或 envelope 已损坏"):
            unwrap_dek(kek, tampered)


if __name__ == "__main__":
    unittest.main()
