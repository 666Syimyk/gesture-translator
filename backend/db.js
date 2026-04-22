import pg from "pg";
import dotenv from "dotenv";

dotenv.config();

const { Pool } = pg;

const poolConfig = {
  host: process.env.DB_HOST || "localhost",
  port: Number(process.env.DB_PORT) || 5432,
  database: process.env.DB_NAME || "gesture_translator",
  user: process.env.DB_USER || "postgres",
};

if (process.env.DB_PASSWORD !== undefined) {
  poolConfig.password = process.env.DB_PASSWORD;
}

export const pool = new Pool(poolConfig);

pool.on("connect", () => {
  console.log("Postgres connected");
});
