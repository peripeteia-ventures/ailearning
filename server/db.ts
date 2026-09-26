import 'dotenv/config';
import pg from 'pg';
export const pool = new pg.Pool({ options: '-c search_path=ailearn,public', max: 10, connectionTimeoutMillis: 5000 });
pool.on('error', error => console.error('PostgreSQL pool:', error.message));
