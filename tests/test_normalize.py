import unittest

from k2dex.tournament_ingest import normalize_name, is_mega_forme


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


class TestIsMegaForme(unittest.TestCase):
    def test_flags_prefix_form(self) -> None:
        for name in ["Mega Blastoise", "Mega Venusaur", "Mega Gardevoir",
                     "Mega Charizard X", "Mega Charizard Y", "Mega Mewtwo X",
                     "Mega Lucario Z"]:
            self.assertTrue(is_mega_forme(name), name)

    def test_flags_hyphen_token_form(self) -> None:
        for name in ["Charizard-Mega-Y", "Charizard-Mega-X", "Blastoise-Mega",
                     "Mewtwo-Mega-X"]:
            self.assertTrue(is_mega_forme(name), name)

    def test_does_not_flag_species_containing_mega_substring(self) -> None:
        # "mega" appears inside these legal species but not as a whole token.
        for name in ["Meganium", "Yanmega"]:
            self.assertFalse(is_mega_forme(name), name)

    def test_does_not_flag_ordinary_species(self) -> None:
        for name in ["Charizard", "Blastoise", "Tapu Koko", "Iron Hands",
                     "Urshifu-Rapid-Strike", "Farfetch'd", "Porygon-Z"]:
            self.assertFalse(is_mega_forme(name), name)


if __name__ == "__main__":
    unittest.main()
