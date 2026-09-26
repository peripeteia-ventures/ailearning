// Real Postgres + a deterministic fake inference endpoint. No Docker, GPU, or provider calls.
import 'dotenv/config';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import pg from 'pg';
const db=new pg.Pool();
const schema=`farmqa_${randomUUID().replaceAll('-','')}`;
assert.match(schema,/^farmqa_[a-f0-9]{32}$/);
let calls=0,mode='complete';
const upstream=createServer(async(req,res)=>{
  if(req.url==='/health'){res.writeHead(200,{'Content-Type':'application/json'});res.end('{"status":"ok"}');return;}
  if(req.url!=='/v1/chat/completions'){res.writeHead(404);res.end();return;}
  calls++;for await(const _ of req){}
  res.writeHead(200,{'Content-Type':'text/event-stream'});
  res.write('data: '+JSON.stringify({choices:[{delta:{content:'A test token. '}}]})+'\n\n');
  if(mode==='complete')res.write('data: '+JSON.stringify({choices:[{delta:{},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n');
  res.end();
});
upstream.listen(0,'127.0.0.1');await new Promise<void>(r=>upstream.once('listening',r));const upstreamAddress=upstream.address();if(!upstreamAddress||typeof upstreamAddress==='string')throw Error('No upstream');
const reservation=createServer();reservation.listen(0,'127.0.0.1');await new Promise<void>(r=>reservation.once('listening',r));const reserved=reservation.address();if(!reserved||typeof reserved==='string')throw Error('No port');const port=reserved.port;await new Promise<void>(r=>reservation.close(()=>r()));
const url=`http://127.0.0.1:${port}`;
const dbUrl=new URL(`postgresql://${process.env.PGHOST??'127.0.0.1'}:${process.env.PGPORT??5432}/${process.env.PGDATABASE??'ai'}`);dbUrl.username=process.env.PGUSER??'postgres';dbUrl.password=process.env.PGPASSWORD??'';dbUrl.searchParams.set('options',`-c search_path=${schema}`);
let child:ReturnType<typeof spawn>|undefined;let output='';
async function request(path:string,body?:unknown,cookie=''){return fetch(url+path,{method:body?'POST':'GET',headers:{Origin:url,...(body?{'Content-Type':'application/json'}:{}),...(cookie?{cookie}:{})},body:body?JSON.stringify(body):undefined});}
try{
  await db.query(`CREATE SCHEMA "${schema}"`);
  const c=await db.connect();try{await c.query(`SET search_path TO "${schema}"`);await c.query(await readFile(new URL('../examples/ai-farm/api/schema.sql',import.meta.url),'utf8'));}finally{c.release();}
  child=spawn(process.execPath,['examples/ai-farm/api/server.js'],{cwd:process.cwd(),windowsHide:true,env:{...process.env,DATABASE_URL:dbUrl.toString(),ADMIN_PASSWORD:'test-only-farm-password',APP_ORIGIN:url,PORT:String(port),HOST:'127.0.0.1',INFERENCE_A_URL:`http://127.0.0.1:${upstreamAddress.port}`,INFERENCE_B_URL:`http://127.0.0.1:${upstreamAddress.port}`,INFERENCE_API_KEY:'fake-test-key',ALLOW_CLOUD:'false'}});
  child.stdout?.on('data',x=>output+=x);child.stderr?.on('data',x=>output+=x);
  let ready=false;for(let i=0;i<40;i++){try{ready=(await fetch(url+'/healthz')).ok;if(ready)break;}catch{}await new Promise(r=>setTimeout(r,100));}assert.ok(ready,output);
  assert.equal((await request('/api/conversations',{})).status,401);
  const login=await request('/api/login',{username:'Admin',password:'test-only-farm-password'});assert.equal(login.status,200);const cookie=login.headers.get('set-cookie')!.split(';')[0];
  const conversation=await(await request('/api/conversations',{},cookie)).json() as {id:string;revision:number};
  const body={requestId:randomUUID(),expectedRevision:0,text:'Hello',provider:'local'};
  const first=await request(`/api/conversations/${conversation.id}/chat`,body,cookie);assert.equal(first.status,200);assert.match(first.headers.get('content-type')!,/text\/event-stream/);const stream=await first.text();assert.ok(stream.includes('event: delta'));assert.ok(stream.includes('event: snapshot'));
  const snap=await(await request(`/api/conversations/${conversation.id}`,undefined,cookie)).json() as any;assert.equal(snap.revision,2);assert.equal(snap.turns.length,1);assert.equal(snap.turns[0].status,'complete');assert.equal(snap.turns[0].assistant_text,'A test token. ');
  const replay=await request(`/api/conversations/${conversation.id}/chat`,body,cookie);assert.equal(replay.status,200);assert.equal((await replay.json() as any).replay,true);assert.equal(calls,1);
  assert.equal((await request(`/api/conversations/${conversation.id}/chat`,{...body,requestId:randomUUID()},cookie)).status,409);
  assert.equal((await request(`/api/conversations/${conversation.id}/chat`,{...body,requestId:randomUUID(),expectedRevision:2,provider:'openai',cloudConsent:true},cookie)).status,403);
  mode='partial';const partial=await request(`/api/conversations/${conversation.id}/chat`,{...body,requestId:randomUUID(),expectedRevision:2},cookie);await partial.text();const partialSnap=await(await request(`/api/conversations/${conversation.id}`,undefined,cookie)).json() as any;assert.equal(partialSnap.revision,4);assert.equal(partialSnap.turns[1].status,'partial');assert.equal(calls,2);
  // Simulate a crashed worker's expired lease; the next authoritative read recovers it.
  await db.query(`UPDATE "${schema}".turns SET status='running',lease_until=now()-interval '1 second' WHERE conversation_id=$1 AND ordinal=$2`,[conversation.id,3]);
  const recovered=await(await request(`/api/conversations/${conversation.id}`,undefined,cookie)).json() as any;assert.equal(recovered.revision,5);assert.equal(recovered.turns[1].error_code,'lease_expired');
  const otherId=randomUUID();await db.query(`INSERT INTO "${schema}".conversations(id,owner_id) VALUES($1,'other-owner')`,[otherId]);assert.equal((await request(`/api/conversations/${otherId}`,undefined,cookie)).status,404);
  console.log('PASS farm teaching scaffold: real Postgres, opaque login, streamed persistence, exact replay, revision conflict, cloud disabled, truncated stream -> partial, expired lease recovery, ownership boundary. Fake inference only.');
}finally{
  if(child&&!child.killed){child.kill();await new Promise<void>(r=>child!.once('exit',()=>r()));}
  await new Promise<void>(r=>upstream.close(()=>r()));
  await db.query(`DROP SCHEMA IF EXISTS "${schema}" CASCADE`);await db.end();
}
