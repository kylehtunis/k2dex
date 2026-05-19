// Sprite components using HTML5 `<object>` for resource fallback.
//
// We tried three other approaches before landing here, all documented
// in rendering_html.py's sprite_img docstring:
//   1. <img onerror=...> — React strips inline handlers
//   2. <script> re-binding — refused execution under dangerouslySetInnerHTML
//   3. layered background-image — transparent sprite pixels let
//      missingno bleed through
//
// `<object>` is the native HTML5 mechanism: when `data` fails to load,
// the browser renders the inner content instead. No JS, no transparency.
// `pointer-events: none` suppresses focus-ring / interactive behavior
// some browsers add to <object>.

import missingnoUrl from "../assets/missingno.svg?url";
import { extractSpecies } from "./format";
import { spriteUrl } from "./sprite-url";

interface SpriteProps {
  /** Vocab string ("Species" or "Species @ Item"). */
  name: string;
  /** Pixel size (both width and height). Default 64. */
  size?: number;
  /** Extra class names on the outer <object>. */
  className?: string;
  /** Add flex-shrink:0 so the sprite doesn't squish inside flex rows. */
  flexShrink?: boolean;
}

/** Sprite as an `<object>` with a missingno fallback child. */
export function SpriteImg({
  name,
  size = 64,
  className,
  flexShrink = false,
}: SpriteProps) {
  const species = extractSpecies(name);
  const src = spriteUrl(name);
  const objectStyle: React.CSSProperties = {
    display: "inline-block",
    width: size,
    height: size,
    pointerEvents: "none",
    border: 0,
    background: "transparent",
    ...(flexShrink ? { flexShrink: 0 } : {}),
  };
  const fallbackStyle: React.CSSProperties = {
    display: "inline-block",
    width: size,
    height: size,
    background: `url('${missingnoUrl}') center/contain no-repeat`,
  };
  return (
    <object
      className={className}
      data={src}
      type="image/png"
      style={objectStyle}
    >
      <span role="img" aria-label={species} style={fallbackStyle} />
    </object>
  );
}

/** Like SpriteImg but with flex-shrink:0 baked in. Used inside slot
 * cards, completion rows, inline pair/swap cells where a flex parent
 * might otherwise squish it. */
export function SpriteBox(props: Omit<SpriteProps, "flexShrink">) {
  return <SpriteImg {...props} flexShrink />;
}
