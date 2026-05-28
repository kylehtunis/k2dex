// Sprite components.
//
// Species sprite uses an <img> with onError → missingno fallback. The
// Streamlit port of this used <object> because Streamlit strips inline
// event handlers, but React's onError works fine here. Using <img> also
// means `new Image()` preloads share the exact cache entry the render
// consumes — no cache misses from header mismatches.
//
// When the vocab string includes "@ Item", a smaller item icon is overlaid
// in the bottom-right corner — pokepast.es-style. The overlay also uses
// <img> with onError → hidden, so unknown items disappear gracefully.

import { useEffect, useState } from "react";
import missingnoUrl from "../assets/missingno.svg?url";
import { extractItem, extractSpecies } from "./format";
import { itemSpriteUrl, spriteUrl } from "./sprite-url";

interface SpriteProps {
  /** Vocab string ("Species" or "Species @ Item"). */
  name: string;
  /** Pixel size (both width and height). Default 64. */
  size?: number;
  /** Extra class names on the outer container. */
  className?: string;
  /** Add flex-shrink:0 so the sprite doesn't squish inside flex rows. */
  flexShrink?: boolean;
}

// Minimum species-sprite size to bother rendering the item overlay.
// Below this it's just visual noise.
const ITEM_OVERLAY_MIN_SIZE = 32;

interface ItemOverlayProps {
  item: string;
  /** Diameter in px. */
  size: number;
}

/** Bottom-right item-icon badge. Hides itself if Showdown doesn't have the
 * icon (gracefully degrades to the species sprite alone). */
function ItemOverlay({ item, size }: ItemOverlayProps) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [item]);
  const src = itemSpriteUrl(item);
  if (!src || failed) return null;
  return (
    <img
      src={src}
      alt={item}
      title={item}
      onError={() => setFailed(true)}
      style={{
        position: "absolute",
        right: -size * 0.15,
        bottom: -size * 0.15,
        width: size,
        height: size,
        pointerEvents: "none",
      }}
    />
  );
}

/** Species sprite as an <img> with onError → missingno fallback. */
function SpeciesImg({
  name,
  size,
}: {
  name: string;
  size: number;
}) {
  const species = extractSpecies(name);
  const src = spriteUrl(name);
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  return (
    <img
      src={failed ? missingnoUrl : src}
      alt={species}
      onError={() => setFailed(true)}
      style={{
        display: "block",
        width: size,
        height: size,
        pointerEvents: "none",
      }}
    />
  );
}

/** Species sprite with an optional bottom-right item-icon overlay for
 * Species @ Item features. */
export function SpriteImg({
  name,
  size = 64,
  className,
  flexShrink = false,
}: SpriteProps) {
  const item = extractItem(name);
  const containerStyle: React.CSSProperties = {
    position: "relative",
    display: "inline-block",
    width: size,
    height: size,
    ...(flexShrink ? { flexShrink: 0 } : {}),
  };
  const overlaySize = Math.round(size * 0.42);
  const showOverlay = item !== null && size >= ITEM_OVERLAY_MIN_SIZE;
  return (
    <span className={className} style={containerStyle} title={name}>
      <SpeciesImg name={name} size={size} />
      {showOverlay && <ItemOverlay item={item!} size={overlaySize} />}
    </span>
  );
}

/** Like SpriteImg but with flex-shrink:0 baked in. Used inside slot
 * cards, completion rows, inline pair/swap cells where a flex parent
 * might otherwise squish it. */
export function SpriteBox(props: Omit<SpriteProps, "flexShrink">) {
  return <SpriteImg {...props} flexShrink />;
}
