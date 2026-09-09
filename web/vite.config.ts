import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies /api + /ws to the backend. The docker stack publishes
// the backend on the HOST at :8001 (compose maps ${BACKEND_PORT:-8001}:8000),
// so that's the default. Override with VITE_BACKEND when running the backend
// elsewhere, e.g. VITE_BACKEND=http://localhost:8000 npm run dev for a bare
// `zevo server` (which listens on :8000 directly).
const BACKEND = process.env.VITE_BACKEND ?? "http://localhost:8001";
const WS_BACKEND = BACKEND.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Keep large, independently cached libraries out of the application
        // entry chunk. This reduces first-load invalidation and avoids shipping
        // the chart/Markdown stacks to routes that do not need them yet.
        manualChunks: {
          "react-vendor": ["react", "react-dom", "react-router-dom"],
          charts: ["recharts"],
          markdown: ["react-markdown", "remark-gfm"],
          icons: ["lucide-react"],
        },
      },
    },
  },
  server: {
    // `npm run dev` runs on 5174 (the script passes --port 5174, which wins
    // over this value): the docker stack owns 5173, where it publishes the
    // BUILT nginx image. Iterating happens on 5174 with HMR; 5173 keeps
    // serving what is actually deployed. The hazard is that the two ports
    // then serve two different builds — 5174 is the uncommitted working
    // copy — so verify on 5173 before calling anything done.
    port: 5174,
    strictPort: true,
    proxy: {
      // WebSocket endpoints under /api/ws/ must come first so the
      // proxy upgrades the connection before the generic /api rule
      // would serve them as plain HTTP.
      "/api/ws": {
        target: WS_BACKEND,
        ws: true,
        changeOrigin: true,
      },
      "/api": {
        target: BACKEND,
        changeOrigin: true,
      },
      "/ws": {
        target: WS_BACKEND,
        ws: true,
      },
    },
  },
});
