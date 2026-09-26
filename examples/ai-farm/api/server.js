// Educational synchronous gateway. See the article and README for production omissions.
import express from 'express';
import pg from 'pg';
import { randomBytes, randomUUID, createHash, timingSafeEqual } from 'node:crypto';
import { fileURLToPath } from 'node:url';
const env=process.env, app=express();
const db=new pg.Pool({connectionString:env.DATABASE_URL,max:12,connectionTimeoutMillis:3000,query_timeout:5000});
const origin=env.APP_ORIGIN||'http://localhost:8088';
if (!env.ADMIN_PASSWORD || env.ADMIN_PASSWORD.length<16) throw Error('ADMIN_PASSWORD must have 16+ characters');
const hash=x=>createHash('sha256').update(x).digest('hex');
const uuid=x=>typeof x==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(x);
const fail=(status,message)=>Object.assign(new Error(message),{status});
let active=0;
// Lab-only, process-wide login throttle. Production needs shared per-principal/IP policy.
let loginWindow=Date.now(),loginAttempts=0;
app.disable('x-powered-by');
app.use(express.json({limit:'16kb'}));
app.use((req,res,next)=>{res.set('Cache-Control','no-store'); if(req.method!=='GET'&&req.headers.origin!==origin)return res.status(403).json({error:'origin_required'}); next();});
app.get('/healthz',async(req,res)=>{await db.query('SELECT 1');res.json({ok:true});});
app.post('/api/login',async(req,res)=>{
 if(Date.now()-loginWindow>=60000){loginWindow=Date.now();loginAttempts=0;}
 if(++loginAttempts>20)throw fail(429,'login_throttled');
 const supplied=Buffer.from(hash(String(req.body?.password||''))), expected=Buffer.from(hash(env.ADMIN_PASSWORD));
 if(req.body?.username!=='Admin'||!timingSafeEqual(supplied,expected))throw fail(401,'invalid_login');
 const token=randomBytes(32).toString('hex');
 await db.query("INSERT INTO sessions VALUES($1,'admin',now()+interval '8 hours')",[hash(token)]);
 res.set('Set-Cookie',`farm_session=${token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800${env.COOKIE_SECURE==='true'?'; Secure':''}`).json({ok:true});
});
app.use('/api',async(req,res,next)=>{
 const token=(req.headers.cookie||'').split(';').map(s=>s.trim()).find(s=>s.startsWith('farm_session='))?.slice(13);
 if(!token)throw fail(401,'login_required');
 const {rows}=await db.query('SELECT owner_id FROM sessions WHERE token_hash=$1 AND expires_at>now()',[hash(token)]);
 if(!rows.length)throw fail(401,'login_required');req.owner=rows[0].owner_id;next();
});
app.post('/api/logout',async(req,res)=>{const token=(req.headers.cookie||'').split(';').map(s=>s.trim()).find(s=>s.startsWith('farm_session='))?.slice(13);await db.query('DELETE FROM sessions WHERE token_hash=$1',[hash(token||'')]);res.set('Set-Cookie','farm_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0').json({ok:true});});
app.post('/api/conversations',async(req,res)=>{const id=randomUUID();await db.query('INSERT INTO conversations(id,owner_id) VALUES($1,$2)',[id,req.owner]);res.status(201).json({id,revision:0});});
// Every mutator locks the conversation first. Recovery and finalization use that same order.
async function locked(id,owner,fn){
 if(!uuid(id))throw fail(400,'invalid_conversation');const c=await db.connect();
 try{await c.query('BEGIN');await c.query("SET LOCAL lock_timeout='3s'");
 const {rows}=await c.query('SELECT * FROM conversations WHERE id=$1 AND owner_id=$2 FOR UPDATE',[id,owner]);
 if(!rows.length)throw fail(404,'not_found');
 const stale=await c.query("UPDATE turns SET status=CASE WHEN assistant_text='' THEN 'error' ELSE 'partial' END,error_code='lease_expired' WHERE conversation_id=$1 AND status='running' AND lease_until<=now() RETURNING request_id",[id]);
 if(stale.rowCount){await c.query('UPDATE conversations SET revision=revision+1 WHERE id=$1',[id]);rows[0].revision++;}
 const result=await fn(c,rows[0]);await c.query('COMMIT');return result;
 }catch(e){await c.query('ROLLBACK');throw e;}finally{c.release();}
}
async function snapshot(c,conv){const {rows}=await c.query('SELECT request_id,ordinal,user_text,assistant_text,provider,status,error_code FROM turns WHERE conversation_id=$1 ORDER BY ordinal',[conv.id]);return {id:conv.id,revision:conv.revision,turns:rows};}
app.get('/api/conversations/:id',async(req,res)=>res.json(await locked(req.params.id,req.owner,snapshot)));
async function chooseLocal(id,signal){
 const nodes=[['a',env.INFERENCE_A_URL],['b',env.INFERENCE_B_URL]].filter(n=>n[1]);
 nodes.sort((a,b)=>hash(id+b[0]).localeCompare(hash(id+a[0])));
 for(const [name,url] of nodes){try{const r=await fetch(`${url.replace(/\/$/,'')}/health`,{headers:{Authorization:`Bearer ${env.INFERENCE_API_KEY||''}`},signal:AbortSignal.any([signal,AbortSignal.timeout(1500)])});if(r.ok)return {name,url};}catch{if(signal.aborted)throw signal.reason;}}
 throw fail(503,'no_healthy_local_model');
}
async function* sse(body){
 const reader=body.getReader(), decoder=new TextDecoder();let buffer='';
 try{while(true){const {done,value}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});buffer=buffer.replace(/\r\n/g,'\n');if(buffer.length>131072)throw Error('upstream_frame_limit');let pos;while((pos=buffer.indexOf('\n\n'))>=0){const frame=buffer.slice(0,pos);buffer=buffer.slice(pos+2);const data=frame.split('\n').filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trimStart()).join('\n');if(data)yield data;}}
 }finally{await reader.cancel().catch(()=>{});reader.releaseLock();}
}
async function upstream(provider,id,messages,signal){
 let url,headers={'Content-Type':'application/json'},body;
 if(provider==='anthropic'){
  url='https://api.anthropic.com/v1/messages';headers['x-api-key']=env.ANTHROPIC_API_KEY;headers['anthropic-version']='2023-06-01';
  body={model:env.ANTHROPIC_MODEL,messages,max_tokens:512,stream:true};
 }else{
  const local=provider==='local';const node=local?await chooseLocal(id,signal):null;
  url=local?`${node.url.replace(/\/$/,'')}/v1/chat/completions`:'https://api.openai.com/v1/chat/completions';
  headers.Authorization=`Bearer ${local?env.INFERENCE_API_KEY:env.OPENAI_API_KEY}`;
  body={model:local?(env.LOCAL_MODEL||'corp-qwen'):env.OPENAI_MODEL,messages,stream:true,...(local?{max_tokens:512}:{max_completion_tokens:512})};
 }
 const r=await fetch(url,{method:'POST',headers,body:JSON.stringify(body),signal});
 if(!r.ok||!r.body)throw Error(`upstream_http_${r.status}`);return r;
}
app.post('/api/conversations/:id/chat',async(req,res)=>{
 const b=req.body||{}, provider=b.provider||'local';
 if(!uuid(b.requestId)||!Number.isInteger(b.expectedRevision)||typeof b.text!=='string'||!b.text.trim()||b.text.length>2000)throw fail(400,'invalid_turn');
 if(!['local','openai','anthropic'].includes(provider))throw fail(400,'invalid_provider');
 if(provider!=='local'&&(env.ALLOW_CLOUD!=='true'||b.cloudConsent!==true))throw fail(403,'cloud_not_approved');
 if(provider==='openai'&&(!env.OPENAI_API_KEY||!env.OPENAI_MODEL))throw fail(503,'openai_unconfigured');
 if(provider==='anthropic'&&(!env.ANTHROPIC_API_KEY||!env.ANTHROPIC_MODEL))throw fail(503,'anthropic_unconfigured');
 const fingerprint=hash(JSON.stringify([b.text,provider,b.expectedRevision,b.cloudConsent===true]));
 // Admission is process-local: two API replicas can admit up to eight total requests.
 if(active>=4)throw fail(429,'replica_busy');active++;
 const ac=new AbortController();let disconnected=false,prepared;
 const close=()=>{if(!res.writableEnded){disconnected=true;ac.abort(Error('client_disconnect'));}};
 res.on('close',close);const timer=setTimeout(()=>ac.abort(Error('generation_timeout')),120000);
 const send=(event,data)=>{if(!res.destroyed){const ok=res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);if(!ok)ac.abort(Error('slow_client'));}};
 try{
 prepared=await locked(req.params.id,req.owner,async(c,conv)=>{
  const old=await c.query('SELECT fingerprint FROM turns WHERE conversation_id=$1 AND request_id=$2',[conv.id,b.requestId]);
  if(old.rowCount){if(old.rows[0].fingerprint!==fingerprint)throw fail(409,'idempotency_key_reused');return {replay:await snapshot(c,conv)};}
  if(conv.revision!==b.expectedRevision)throw fail(409,'revision_conflict');
  const history=await c.query('SELECT * FROM turns WHERE conversation_id=$1 ORDER BY ordinal',[conv.id]);
  if(history.rows.some(t=>t.status==='running'))throw fail(409,'generation_active');
  const messages=history.rows.filter(t=>t.status==='complete').flatMap(t=>[{role:'user',content:t.user_text},{role:'assistant',content:t.assistant_text}]);messages.push({role:'user',content:b.text});
  if(messages.reduce((n,m)=>n+m.content.length,0)>6000)throw fail(413,'history_limit_start_new_conversation');
  await c.query("INSERT INTO turns(conversation_id,request_id,fingerprint,ordinal,user_text,provider,status,lease_until) VALUES($1,$2,$3,$4,$5,$6,'running',now()+interval '150 seconds')",[conv.id,b.requestId,fingerprint,conv.revision+1,b.text,provider]);
  await c.query('UPDATE conversations SET revision=revision+1 WHERE id=$1',[conv.id]);return {messages,revision:conv.revision+1};
 });
 if(prepared.replay){res.json({replay:true,...prepared.replay});return;}
 res.set({'Content-Type':'text/event-stream','X-Accel-Buffering':'no'});res.flushHeaders();send('accepted',{requestId:b.requestId,revision:prepared.revision});
 let output='',status='complete',code=null,terminal=false;
 try{
  const r=await upstream(provider,req.params.id,prepared.messages,ac.signal);
  for await(const data of sse(r.body)){
   if(data==='[DONE]'){terminal=true;break;}const e=JSON.parse(data);
   if(e.error||e.type==='error')throw Error('upstream_stream_error');
   if(e.type==='message_stop'){terminal=true;break;}
   const delta=provider==='anthropic'?(e.type==='content_block_delta'&&e.delta?.type==='text_delta'?e.delta.text:''):(e.choices?.[0]?.delta?.content||'');
   if(typeof delta!=='string')throw Error('unsupported_delta');if(!delta)continue;
   if(output.length+delta.length>16000)throw Error('output_limit');output+=delta;
   // Durable-before-visible is intentionally expensive but avoids losing displayed text on crash.
   const saved=await db.query("UPDATE turns SET assistant_text=$3 WHERE conversation_id=$1 AND request_id=$2 AND status='running' AND lease_until>now()",[req.params.id,b.requestId,output]);
   if(!saved.rowCount)throw Error('lease_lost');send('delta',{text:delta});if(ac.signal.aborted)throw ac.signal.reason;
  }
  if(!terminal)throw Error('truncated_stream');
 }catch(e){status=disconnected?'canceled':output?'partial':'error';code=ac.signal.aborted?(disconnected?'client_disconnect':'generation_aborted'):'upstream_failed';}
 const end=await locked(req.params.id,req.owner,async(c,conv)=>{
  const done=await c.query("UPDATE turns SET status=$3,error_code=$4 WHERE conversation_id=$1 AND request_id=$2 AND status='running' AND lease_until>now() RETURNING request_id",[conv.id,b.requestId,status,code]);
  if(done.rowCount){await c.query('UPDATE conversations SET revision=revision+1 WHERE id=$1',[conv.id]);conv.revision++;}return snapshot(c,conv);
 });send('snapshot',end);res.end();
 }catch(e){if(res.headersSent){send('reconcile',{error:'fetch_snapshot_after_lease_expiry'});res.end();}else throw e;}
 finally{clearTimeout(timer);res.off('close',close);ac.abort();active--;}
});
app.use(express.static(fileURLToPath(new URL('./public',import.meta.url))));
app.use((err,req,res,next)=>{console.error(JSON.stringify({event:'request_error',code:err.status||500,path:req.path}));if(!res.headersSent)res.status(err.status||503).json({error:err.status?err.message:'temporarily_unavailable'});else res.end();});
app.listen(Number(env.PORT||3000),env.HOST||'0.0.0.0');
