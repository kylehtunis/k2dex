import { describe, expect, it } from "vitest";
import {
  extractAbility,
  extractItem,
  extractSpecies,
  formatPair,
  formatTriple,
} from "../format";

describe("extractSpecies / extractItem / extractAbility", () => {
  it("parses a bare species (no item, no ability track)", () => {
    expect(extractSpecies("Incineroar")).toBe("Incineroar");
    expect(extractItem("Incineroar")).toBeNull();
    expect(extractAbility("Incineroar")).toBeNull();
  });

  it("parses species + item (v4 item-only vocab, current committed artifacts)", () => {
    const f = "Incineroar @ Sitrus Berry";
    expect(extractSpecies(f)).toBe("Incineroar");
    expect(extractItem(f)).toBe("Sitrus Berry");
    expect(extractAbility(f)).toBeNull();
  });

  it("parses species + item + ability", () => {
    const f = "Incineroar @ Sitrus Berry (Intimidate)";
    expect(extractSpecies(f)).toBe("Incineroar");
    expect(extractItem(f)).toBe("Sitrus Berry");
    expect(extractAbility(f)).toBe("Intimidate");
  });

  it("parses an itemless species + ability", () => {
    const f = "Talonflame (Gale Wings)";
    expect(extractSpecies(f)).toBe("Talonflame");
    expect(extractItem(f)).toBeNull();
    expect(extractAbility(f)).toBe("Gale Wings");
  });

  it("handles multi-word abilities with hyphens and spaces", () => {
    const f = "Kingambit @ Black Glasses (Supreme Overlord)";
    expect(extractSpecies(f)).toBe("Kingambit");
    expect(extractItem(f)).toBe("Black Glasses");
    expect(extractAbility(f)).toBe("Supreme Overlord");

    const f2 = "Calyrex-Shadow @ Life Orb (As One)";
    expect(extractSpecies(f2)).toBe("Calyrex-Shadow");
    expect(extractItem(f2)).toBe("Life Orb");
    expect(extractAbility(f2)).toBe("As One");
  });
});

describe("formatPair / formatTriple", () => {
  it("formatPair round-trips through extractSpecies/extractItem", () => {
    expect(formatPair("Incineroar", "Sitrus Berry")).toBe("Incineroar @ Sitrus Berry");
    expect(formatPair("Incineroar", null)).toBe("Incineroar");
  });

  it("formatTriple appends the ability parenthetical after the item", () => {
    expect(formatTriple("Incineroar", "Sitrus Berry", "Intimidate")).toBe(
      "Incineroar @ Sitrus Berry (Intimidate)",
    );
  });

  it("formatTriple drops the item segment for itemless builds (item null or the literal None)", () => {
    expect(formatTriple("Talonflame", null, "Gale Wings")).toBe("Talonflame (Gale Wings)");
    expect(formatTriple("Talonflame", "None", "Gale Wings")).toBe("Talonflame (Gale Wings)");
  });

  it("round-trips formatTriple through the extractors", () => {
    const built = formatTriple("Kingambit", "Black Glasses", "Supreme Overlord");
    expect(extractSpecies(built)).toBe("Kingambit");
    expect(extractItem(built)).toBe("Black Glasses");
    expect(extractAbility(built)).toBe("Supreme Overlord");
  });
});
