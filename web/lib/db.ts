import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import * as schema from "./schema";

// Placeholder keeps imports from throwing before DATABASE_URL is set; real queries
// fail until a valid Neon connection string is in .env.
const sql = neon(process.env.DATABASE_URL || "postgres://user:pass@localhost/db");

export const db = drizzle(sql, { schema });
