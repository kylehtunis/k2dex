import { useState } from "react";
import { ANNOUNCEMENT } from "../announcement";

const STORAGE_PREFIX = "k2dex-dismissed-announcement:";

export function AnnouncementBanner() {
  const [dismissed, setDismissed] = useState(() => {
    if (!ANNOUNCEMENT) return true;
    try {
      return localStorage.getItem(STORAGE_PREFIX + ANNOUNCEMENT) === "1";
    } catch {
      return false;
    }
  });

  if (!ANNOUNCEMENT || dismissed) return null;

  const dismiss = () => {
    setDismissed(true);
    try {
      localStorage.setItem(STORAGE_PREFIX + ANNOUNCEMENT, "1");
    } catch {
      // storage full or blocked — banner won't persist but still dismisses
    }
  };

  return (
    <div className="lab-announcement">
      <span className="lab-announcement-text" dangerouslySetInnerHTML={{ __html: ANNOUNCEMENT }} />
      <button
        className="lab-announcement-close"
        onClick={dismiss}
        aria-label="Dismiss announcement"
      >
        &times;
      </button>
    </div>
  );
}
