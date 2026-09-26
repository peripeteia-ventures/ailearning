import {readFile,writeFile,unlink} from 'node:fs/promises';
import {randomUUID} from 'node:crypto';
import {pool} from '../server/db.ts';
import {hashPassword} from '../server/password.ts';
const record=new URL('../work/qa-user.json',import.meta.url);
try {
  if(process.argv[2]==='remove') {const {id,username}=JSON.parse(await readFile(record,'utf8'));await pool.query('DELETE FROM users WHERE id=$1 AND username=$2',[id,username]);await unlink(record);console.log('Disposable browser QA user removed.');}
  else {const username=`qa-browser-${randomUUID().slice(0,8)}`;const row=(await pool.query('INSERT INTO users(username,password_hash) VALUES($1,$2) RETURNING id,username',[username,hashPassword('qa-local-123')])).rows[0];await writeFile(record,JSON.stringify(row));console.log(JSON.stringify({...row,password:'qa-local-123'}));}
}finally{await pool.end();}
