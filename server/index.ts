import { createApp } from './app.ts';
import { pool } from './db.ts';
await pool.query('SELECT 1 FROM ailearn.schema_migrations LIMIT 1');
const port=Number(process.env.PORT??3002);
const server=createApp().listen(port,'127.0.0.1',()=>console.log(`Latent ready at http://127.0.0.1:${port}`));
async function shutdown(){server.close(async()=>{await pool.end();process.exit(0);});}
process.on('SIGINT',shutdown); process.on('SIGTERM',shutdown);
