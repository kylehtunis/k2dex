import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles/index.css";

// GitHub Pages SPA fallback: 404.html redirects here with ?route=/original-path.
// Restore the clean URL before React mounts so BrowserRouter sees the right path.
const params = new URLSearchParams(location.search);
const route = params.get("route");
if (route) {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  history.replaceState(null, "", base + route + location.hash);
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
