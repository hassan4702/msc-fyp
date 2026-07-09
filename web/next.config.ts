import type { NextConfig } from "next";

// Proxy /api/* to the FastAPI emotion backend so the browser stays same-origin (no CORS).
// Override the target with API_URL when the backend runs elsewhere.
const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
};

export default nextConfig;
