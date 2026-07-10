import type { NextConfig } from "next";

// Proxy /backend/* to the FastAPI emotion backend (same-origin, no CORS).
// Kept off /api/* so it never collides with Next's own /api/auth (Better Auth).
const API_URL = process.env.API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/backend/:path*", destination: `${API_URL}/:path*` }];
  },
};

export default nextConfig;
