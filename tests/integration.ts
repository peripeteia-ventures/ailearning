import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {createApp} from '../server/app.ts';
import {pool} from '../server/db.ts';
import {hashPassword} from '../server/password.ts';
import {setup} from '../server/setup.ts';
const ids:number[]=[];const server=createApp().listen(0,'127.0.0.1');await new Promise<void>(r=>server.once('listening',r));const address=server.address();if(!address||typeof address==='string')throw new Error('No test listener');const base=`http://127.0.0.1:${address.port}`;
type Result={status:number;data:any;cookie:string};
async function request(route:string,body?:unknown,cookie='',method=body?'POST':'GET',origin?:string):Promise<Result>{const r=await fetch(base+'/api'+route,{method,headers:{...(body?{'Content-Type':'application/json'}:{}),...(cookie?{cookie}:{}),...(origin?{origin}:{})},body:body?JSON.stringify(body):undefined});return{status:r.status,data:await r.json(),cookie:r.headers.get('set-cookie')?.split(';')[0]??''};}
try{
  await setup();
  const names=[`qa-${randomUUID()}`,`qa-${randomUUID()}`];
  for(const name of names)ids.push((await pool.query('INSERT INTO users(username,password_hash) VALUES($1,$2) RETURNING id',[name,hashPassword('qa-local-123')])).rows[0].id);
  assert.equal((await request('/health')).status,200);
  assert.equal((await request('/catalog')).status,401);
  assert.equal((await request('/login',{username:names[0],password:'wrong'})).status,401);
  assert.equal((await request('/login',{username:names[0],password:'qa-local-123'},'',undefined,'https://evil.example')).status,403);
  const login=await request('/login',{username:names[0],password:'qa-local-123'},'',undefined,'https://momentdesktop.tail01307d.ts.net:10001');assert.equal(login.status,200);assert.ok(login.cookie);const cookie=login.cookie;
  const other=(await request('/login',{username:names[1],password:'qa-local-123'})).cookie;
  const catalog=await request('/catalog',undefined,cookie);assert.equal(catalog.status,200);assert.ok(catalog.data.articles.length>=12);
  const slug='transformer-foundations';const path=`/articles/${slug}`;
  assert.equal((await request('/articles/missing',undefined,cookie)).status,404);
  assert.equal((await request(`${path}/progress`,{isRead:true,bookmarked:true},cookie,'PATCH')).status,200);
  assert.equal((await request('/review',undefined,cookie)).data.cards.length,0);
  assert.equal((await request(path,undefined,other)).data.progress.isRead,false);
  await request(`${path}/enroll`,{},cookie);await request(`${path}/enroll`,{},cookie);
  const queue=(await request('/review?newLimit=50',undefined,cookie)).data.cards;const lesson=(await request(path,undefined,cookie)).data;assert.equal(queue.length,lesson.cards.length);
  const card=queue[0];const payload={cardId:card.id,quality:4,version:0,requestId:randomUUID(),responseMs:1000,kind:'scheduled'};
  const first=await request('/reviews',payload,cookie);assert.equal(first.status,200);assert.equal(first.data.state.attempts,1);
  const repeat=await request('/reviews',payload,cookie);assert.equal(repeat.status,200);assert.equal(repeat.data.replayed,true);assert.deepEqual(repeat.data.state,first.data.state);
  assert.equal((await request('/reviews',{...payload,quality:5},cookie)).status,409);
  assert.equal((await request('/reviews',{...payload,requestId:randomUUID()},cookie)).status,409);
  assert.equal((await request('/reviews',{...payload,version:1,requestId:randomUUID()},cookie)).status,409);
  const practice=await request('/reviews',{...payload,version:1,kind:'practice',quality:2,requestId:randomUUID()},cookie);assert.equal(practice.status,200);assert.equal(practice.data.state.due,first.data.state.due);assert.equal(practice.data.state.attempts,1);assert.equal(practice.data.state.practice,1);
  assert.equal((await request('/reviews',{...payload,requestId:randomUUID()},other)).status,404);
  const race=await Promise.all([0,1].map(()=>request('/reviews',{...payload,cardId:queue[1].id,requestId:randomUUID()},cookie)));assert.deepEqual(race.map(x=>x.status).sort(),[200,409]);
  assert.equal((await request('/reviews',{...payload,quality:8,requestId:randomUUID()},cookie)).status,400);
  const stats=(await request('/stats',undefined,cookie)).data;assert.equal(stats.attempts,2);assert.equal(stats.correct,2);assert.equal(stats.read,1);assert.equal(stats.enrolled,1);assert.equal(stats.recent.length,3);
  const before=(await request(path,undefined,cookie)).data;await setup();const after=(await request(path,undefined,cookie)).data;assert.deepEqual(after.progress,before.progress);assert.deepEqual(after.cards,before.cards);
  const verified=(await request('/catalog',undefined,cookie)).data.articles;
  for(const a of verified){const data=(await request(`/articles/${a.slug}`,undefined,cookie)).data;assert.equal(data.article.slug,a.slug);assert.equal(data.cards.length,data.article.flashcards.length);assert.ok(data.article.sections.length>=6);}
  await request('/logout',{},cookie);assert.equal((await request('/me',undefined,cookie)).status,401);
  console.log(`PASS: auth/origins, isolated progress, enrollment, scheduling, retry idempotency, conflicts, concurrent writes, practice, reseed preservation, all ${verified.length} article payloads, logout. Disposable users removed.`);
}finally{for(const id of ids)await pool.query('DELETE FROM users WHERE id=$1',[id]);await new Promise<void>(r=>server.close(()=>r()));await pool.end();}
