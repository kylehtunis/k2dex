import unittest

from k2dex.tournament_ingest import normalize_name, strip_mega_prefix


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


class TestStripMegaPrefix(unittest.TestCase):
    def test_strips_simple_mega(self) -> None:
        self.assertEqual(strip_mega_prefix("Mega Blastoise"), "Blastoise")
        self.assertEqual(strip_mega_prefix("Mega Venusaur"), "Venusaur")
        self.assertEqual(strip_mega_prefix("Mega Gardevoir"), "Gardevoir")

    def test_strips_x_y_z_suffix_when_mega_prefix_present(self) -> None:
        self.assertEqual(strip_mega_prefix("Mega Charizard X"), "Charizard")
        self.assertEqual(strip_mega_prefix("Mega Charizard Y"), "Charizard")
        self.assertEqual(strip_mega_prefix("Mega Mewtwo X"), "Mewtwo")
        self.assertEqual(strip_mega_prefix("Mega Mewtwo Y"), "Mewtwo")
        # Z is included for future-proofing.
        self.assertEqual(strip_mega_prefix("Mega Lucario Z"), "Lucario")

    def test_passthrough_for_non_mega(self) -> None:
        # Non-mega species are unchanged.
        for name in ["Charizard", "Blastoise", "Tapu Koko", "Iron Hands",
                     "Urshifu-Rapid-Strike", "Farfetch'd", "Porygon-Z"]:
            self.assertEqual(strip_mega_prefix(name), name)

    def test_does_not_strip_x_y_z_without_mega(self) -> None:
        # The trailing forme letter is only stripped when 'Mega ' prefix is
        # present, so unusual species ending in stray letters round-trip.
        self.assertEqual(strip_mega_prefix("Porygon-Z"), "Porygon-Z")
        self.assertEqual(strip_mega_prefix("Charizard X"), "Charizard X")

    def test_idempotent(self) -> None:
        for name in ["Mega Blastoise", "Mega Charizard Y", "Charizard",
                     "Tapu Koko", "Mega Lucario Z"]:
            self.assertEqual(
                strip_mega_prefix(strip_mega_prefix(name)),
                strip_mega_prefix(name),
            )


if __name__ == "__main__":
    unittest.main()
