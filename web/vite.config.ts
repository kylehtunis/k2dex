import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Production serves at the root of the custom domain (k2dex.kyletunis.com),
// so the deploy workflow sets VITE_BASE_PATH=/. Dev / local-preview default to /.
// Override with VITE_BASE_PATH for non-standard deploys.
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE_PATH ?? "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    // dist/.vite/manifest.json: source module -> emitted chunk files. The
    // prerender reads it to inject per-route <link rel="modulepreload"> tags
    // for lazy article chunks.
    manifest: true,
  },
  server: {
    port: 5173,
  },
  // The prerender script (vite-node) imports react-router-dom/server while the
  // app tree imports react-router-dom; if the packages stay external the two
  // entries resolve to different module instances and the Router context
  // breaks. noExternal pipes them through one transform, one instance.
  ssr: {
    noExternal: ["react-router-dom", "react-router", "@remix-run/router"],
  },
  test: {
    globals: true,
    environment: "node",
  },
});
