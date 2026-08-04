// Generic portal modal primitive. No prior modal existed in the app (closest
// precedents are the .lab-model-picker-panel dropdown and .lab-stat-pop
// tooltip), so this is the single reusable shell: backdrop, ESC-to-close,
// click-outside-to-close, focus trap + focus restore, and body scroll lock.

import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";

export interface ModalProps {
  onClose: () => void;
  /** id of the element labelling the dialog (for aria-labelledby). */
  labelledBy?: string;
  /** "modal" (default): centered overlay, backdrop, scroll-lock + focus-trap.
   * "dock": fixed right-hand inspector panel — page stays interactive, no
   * backdrop / scroll-lock / focus-trap; only ESC-to-close is shared. */
  variant?: "modal" | "dock";
  children: ReactNode;
}

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function Modal({ onClose, labelledBy, variant = "modal", children }: ModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const restoreFocusTo = useRef<HTMLElement | null>(null);

  // Lock body scroll + capture/restore focus — modal variant only. The dock
  // is a non-blocking inspector, so it must leave the page scrollable/focusable.
  useEffect(() => {
    if (variant !== "modal") return;
    restoreFocusTo.current = document.activeElement as HTMLElement | null;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialogRef.current?.focus();
    return () => {
      document.body.style.overflow = prevOverflow;
      restoreFocusTo.current?.focus?.();
    };
  }, [variant]);

  // Dock variant: reserve the right gutter so the centered page content slides
  // left (the shift is emergent from .lab-container's margin:auto).
  useEffect(() => {
    if (variant !== "dock") return;
    document.body.classList.add("feature-docked");
    return () => document.body.classList.remove("feature-docked");
  }, [variant]);

  // ESC closes (both variants). Tab focus-trap is modal-only — the dock lets
  // focus flow back out to the page. Capture phase so a partner-drill-through
  // inside the panel doesn't see a stray Escape.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        // An open dropdown owns Escape: it should close the menu, not the whole
        // panel. This listener is on `document` in capture phase, so without
        // the bail it fires before react-select's own handler ever runs (React
        // delegates from the root container, a descendant of document) and the
        // panel closes with the menu still logically open. react-select only
        // renders `__menu` while the menu is open, in a portal.
        if (document.querySelector(".lab-select__menu")) return;
        e.stopPropagation();
        onClose();
        return;
      }
      if (variant !== "modal" || e.key !== "Tab") return;
      const nodes = dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE);
      if (!nodes || nodes.length === 0) {
        e.preventDefault();
        dialogRef.current?.focus();
        return;
      }
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [onClose, variant]);

  if (variant === "dock") {
    return createPortal(
      <div
        ref={dialogRef}
        className="lab-feature-dock"
        role="dialog"
        aria-label="Feature detail"
        aria-labelledby={labelledBy}
        tabIndex={-1}
      >
        {children}
      </div>,
      document.body,
    );
  }

  // mousedown (not click) on the backdrop so a text selection that drifts onto
  // the backdrop on mouseup doesn't dismiss the dialog.
  return createPortal(
    <div className="lab-modal-backdrop" onMouseDown={onClose}>
      <div
        ref={dialogRef}
        className="lab-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        tabIndex={-1}
        onMouseDown={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>,
    document.body,
  );
}
