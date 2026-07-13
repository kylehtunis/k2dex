// react-select multi-select keyed by vocab index. Used for both the
// "pin" (vocab string -> idx) and "exclude" (species string ->
// list of idx) constraints on the completer / analysis pages.


import Select, { type MultiValue } from "react-select";
import type { IsingModel } from "../sampler/types";

export interface VocabOption {
  /** Display label (vocab string). */
  label: string;
  /** Vocab index. */
  value: number;
}

interface VocabSelectProps {
  /** Available options. Sort by caller (typically by marginal). */
  options: VocabOption[];
  /** Currently selected vocab indices. */
  value: readonly number[];
  onChange: (next: number[]) => void;
  /** Disable selecting more than N (TEAM_SIZE for pinned). */
  maxSelections?: number;
  placeholder?: string;
  /** Accessible label for screen readers. */
  ariaLabel?: string;
}

export function VocabSelect({
  options,
  value,
  onChange,
  maxSelections,
  placeholder,
  ariaLabel,
}: VocabSelectProps) {
  const valueOptions = value
    .map((i) => options.find((o) => o.value === i))
    .filter((o): o is VocabOption => o !== undefined);
  const atLimit = maxSelections !== undefined && value.length >= maxSelections;
  return (
    <Select
      classNamePrefix="lab-select"
      isMulti
      options={atLimit ? [] : options}
      value={valueOptions}
      onChange={(sel: MultiValue<VocabOption>) =>
        onChange(sel.map((o) => o.value))
      }
      placeholder={placeholder}
      aria-label={ariaLabel}
      noOptionsMessage={() =>
        atLimit ? `Max ${maxSelections} selected` : "No matches"
      }
      menuPortalTarget={document.body}
      styles={{
        menuPortal: (base) => ({ ...base, zIndex: 9999 }),
      }}
    />
  );
}

/** Like VocabSelect but the option values are species name strings
 * (used for the species-level exclude multiselect). */
interface SpeciesSelectProps {
  options: Array<{ label: string; value: string }>;
  value: readonly string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
  ariaLabel?: string;
}

export function SpeciesSelect({
  options,
  value,
  onChange,
  placeholder,
  ariaLabel,
}: SpeciesSelectProps) {
  const valueOptions = value
    .map((s) => options.find((o) => o.value === s))
    .filter((o): o is { label: string; value: string } => o !== undefined);
  return (
    <Select
      classNamePrefix="lab-select"
      isMulti
      options={options}
      value={valueOptions}
      onChange={(sel) => onChange((sel as MultiValue<{ value: string }>).map((o) => o.value))}
      placeholder={placeholder}
      aria-label={ariaLabel}
      menuPortalTarget={document.body}
      styles={{
        menuPortal: (base) => ({ ...base, zIndex: 9999 }),
      }}
    />
  );
}

/** Convenience: build sorted VocabOption list for the pin select. */
export function vocabOptions(model: IsingModel): VocabOption[] {
  const idxs = Array.from({ length: model.V }, (_, i) => i);
  idxs.sort((a, b) => model.m[b] - model.m[a]);
  return idxs.map((i) => ({ label: model.vocab[i], value: i }));
}

/** Convenience: build sorted species options for the exclude select. */
export function speciesOptions(model: IsingModel): Array<{ label: string; value: string }> {
  const popByName = new Map<string, number>();
  for (let i = 0; i < model.V; i++) {
    const sp = model.speciesOf[i];
    popByName.set(sp, (popByName.get(sp) ?? 0) + model.m[i]);
  }
  const names = Array.from(popByName.keys());
  names.sort((a, b) => (popByName.get(b) ?? 0) - (popByName.get(a) ?? 0));
  return names.map((s) => ({ label: s, value: s }));
}
