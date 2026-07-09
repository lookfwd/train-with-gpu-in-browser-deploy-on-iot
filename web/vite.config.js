import { defineConfig } from "vite";

// Relative base so the built dist/ works whether it's served from a domain root
// or a subpath (e.g. GitHub Pages project sites).
export default defineConfig({
  base: "./",
  server: { port: 5173, host: true },
  build: { target: "es2022", outDir: "dist" },
  // mqtt.js pulls in a few node built-ins; its browser build is fine, but pre-bundling
  // it keeps dev fast and avoids CJS/ESM interop hiccups.
  optimizeDeps: { include: ["mqtt"] },
});
