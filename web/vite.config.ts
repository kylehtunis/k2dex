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
  },
  server: {
    port: 5173,
  },
  test: {
    globals: true,
    environment: "node",
  },
});
