import unittest

from limitless_ingest import normalize_name


class TestNormalizeName(unittest.TestCase):
    def test_collapses_case_variants(self) -> None:
        for raw in ["Sitrus Berry", "Sitrus berry", "sitrus berry", "SITRUS BERRY"]:
            self.assertEqual(normalize_name(raw), "Sitrus Berry")

    def test_preserves_hyphenated_formes(self) -> None:
        self.assertEqual(normalize_name("charizard-mega-y"), "Charizard-Mega-Y")
        self.assertEqual(normalize_name("Porygon-z"), "Porygon-Z")
        self.assertEqual(normalize_name("urshifu-rapid-strike"), "Urshifu-Rapid-Strike")

    def test_preserves_apostrophes(self) -> None:
        self.assertEqual(normalize_name("farfetch'd"), "Farfetch'd")
        self.assertEqual(normalize_name("FARFETCH'D"), "Farfetch'd")

    def test_handles_multiword(self) -> None:
        self.assertEqual(normalize_name("tapu koko"), "Tapu Koko")
        self.assertEqual(normalize_name("CHOICE SCARF"), "Choice Scarf")
        self.assertEqual(normalize_name("flutter mane"), "Flutter Mane")

    def test_returns_none_for_empty(self) -> None:
        self.assertIsNone(normalize_name(None))
        self.assertIsNone(normalize_name(""))
        self.assertIsNone(normalize_name("   "))

    def test_idempotent(self) -> None:
        for raw in ["Sitrus Berry", "Tapu Koko", "Porygon-Z", "Farfetch'd",
                    "Charizard-Mega-Y", "Iron Hands"]:
            self.assertEqual(normalize_name(normalize_name(raw)), raw)


if __name__ == "__main__":
    unittest.main()
