import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev server on :5173. we proxy /api -> the FastAPI backend on :8000 so the
// frontend can call "/api/predict" without any CORS / hardcoded-port fuss.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
