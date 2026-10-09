import express, { type Request, type Response, type NextFunction } from 'express';
import { randomBytes } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';
import { pool } from './db.ts';
import { tokenHash, verifyPassword } from './password.ts';
import { initialState, schedule, type Schedule } from './scheduler.ts';
import { categories } from '../shared/catalog.ts';

declare global { namespace Express { interface Request { user?: { id: number; username: string } } } }
class HttpError extends Error { status:number; constructor(status: number, message: string) { super(message); this.status=status; } }
const requireValue = (value: unknown, status: number, message: string) => { if (!value) throw new HttpError(status,message); };
const cookieName = 'latent_session';
function sessionToken(req: Request) { return req.headers.cookie?.split(';').map(x=>x.trim()).find(x=>x.startsWith(`${cookieName}=`))?.slice(cookieName.length+1) ?? ''; }
export function createApp() {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy','loopback');
  app.use(express.json({ limit: '32kb' }));
  app.use((_req,res,next)=>{ res.setHeader('X-Content-Type-Options','nosniff'); res.setHeader('Referrer-Policy','same-origin'); res.setHeader('X-Frame-Options','DENY'); next(); });
  const allowed = new Set((process.env.ALLOWED_ORIGINS ?? 'http://127.0.0.1:3002,http://localhost:3002,http://127.0.0.1:5174').split(',').map(x=>x.trim()));
  app.use('/api',(req,res,next)=>{ res.setHeader('Cache-Control','no-store'); if (req.headers.origin && !allowed.has(req.headers.origin)) return next(new HttpError(403,'This browser origin is not allowed.')); next(); });
  app.get('/api/health',async(_req,res)=>{ await pool.query('SELECT 1'); res.json({ok:true,app:'latent',database:'ai',version:'1.0.0'}); });
  const loginAttempts = new Map<string,{count:number;until:number}>();
  app.post('/api/login',async(req,res)=>{
    const {username,password} = z.object({username:z.string().min(1).max(100),password:z.string().min(1).max(200)}).parse(req.body);
    const key = req.ip ?? 'local'; const attempt = loginAttempts.get(key);
    if (attempt && attempt.until > Date.now() && attempt.count >= 15) throw new HttpError(429,'Too many attempts. Try again in a few minutes.');
    const user = (await pool.query('SELECT id,username,password_hash FROM users WHERE lower(username)=lower($1)',[username])).rows[0];
    if (!user || !verifyPassword(password,user.password_hash)) {
      loginAttempts.set(key,{count:attempt && attempt.until>Date.now()?attempt.count+1:1,until:Date.now()+300000});
      throw new HttpError(401,'Username or password is incorrect.');
    }
    loginAttempts.delete(key);
    const token = randomBytes(32).toString('hex');
    await pool.query('INSERT INTO sessions(token_hash,user_id,expires_at) VALUES($1,$2,now()+interval \'30 days\')',[tokenHash(token),user.id]);
    res.cookie(cookieName,token,{httpOnly:true,sameSite:'lax',secure:req.secure,path:'/',maxAge:30*86400000});
    res.json({id:user.id,username:user.username});
  });
  app.post('/api/logout',async(req,res)=>{ await pool.query('DELETE FROM sessions WHERE token_hash=$1',[tokenHash(sessionToken(req))]); res.clearCookie(cookieName,{path:'/',httpOnly:true,sameSite:'lax',secure:req.secure}); res.json({ok:true}); });
  app.use('/api',async(req,_res,next)=>{
    const token = sessionToken(req);
    if (!/^[a-f0-9]{64}$/.test(token)) throw new HttpError(401,'Please sign in.');
    const user = (await pool.query('SELECT u.id,u.username FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=$1 AND s.expires_at>now()',[tokenHash(token)])).rows[0];
    requireValue(user,401,'Your session expired. Please sign in.'); req.user=user; next();
  });
  app.get('/api/me',(req,res)=>res.json(req.user));
  app.get('/api/catalog',async(req,res)=>{
    const rows = (await pool.query(`SELECT a.slug,a.category,a.position,a.content->>'title' title,a.content->>'summary' summary,a.content->>'difficulty' difficulty,(a.content->>'minutes')::int minutes,jsonb_array_length(a.content->'flashcards') "cardCount",jsonb_array_length(a.content->'sections') "sectionCount",(SELECT count(*)::int FROM jsonb_array_elements(a.content->'sections') s WHERE s ? 'diagram') "diagramCount",COALESCE(ua.is_read,false) "isRead",COALESCE(ua.bookmarked,false) bookmarked,ua.enrolled_at "enrolledAt",COALESCE(c.due,0)::int due,COALESCE(c.fresh,0)::int fresh,COALESCE(c.established,0)::int established FROM articles a LEFT JOIN user_articles ua ON ua.article_slug=a.slug AND ua.user_id=$1 LEFT JOIN LATERAL(SELECT count(*) FILTER(WHERE (state->>'attempts')::int>0 AND due_at<=now()) due,count(*) FILTER(WHERE (state->>'attempts')::int=0) fresh,count(*) FILTER(WHERE (state->>'repetitions')::int>=3 AND (state->>'interval')::int>=21) established FROM user_cards WHERE user_id=$1 AND article_slug=a.slug)c ON true ORDER BY a.position`,[req.user!.id])).rows;
    res.json({categories,articles:rows});
  });
  app.get('/api/articles/:slug',async(req,res)=>{
    const row=(await pool.query('SELECT content FROM articles WHERE slug=$1',[req.params.slug])).rows[0]; requireValue(row,404,'Article not found.');
    const progress=(await pool.query('SELECT is_read "isRead",bookmarked,enrolled_at "enrolledAt" FROM user_articles WHERE user_id=$1 AND article_slug=$2',[req.user!.id,req.params.slug])).rows[0];
    const cards=(await pool.query('SELECT c.id,c.front,c.back,uc.state FROM cards c LEFT JOIN user_cards uc ON uc.card_id=c.id AND uc.user_id=$1 WHERE c.article_slug=$2 AND NOT c.retired ORDER BY c.position',[req.user!.id,req.params.slug])).rows;
    res.json({article:row.content,progress:progress??{isRead:false,bookmarked:false,enrolledAt:null},cards});
  });
  app.patch('/api/articles/:slug/progress',async(req,res)=>{
    const body=z.object({isRead:z.boolean().optional(),bookmarked:z.boolean().optional()}).strict().parse(req.body);
    requireValue((await pool.query('SELECT 1 FROM articles WHERE slug=$1',[req.params.slug])).rowCount,404,'Article not found.');
    const row=(await pool.query(`INSERT INTO user_articles(user_id,article_slug,is_read,bookmarked) VALUES($1,$2,COALESCE($3,false),COALESCE($4,false)) ON CONFLICT(user_id,article_slug) DO UPDATE SET is_read=COALESCE($3,user_articles.is_read),bookmarked=COALESCE($4,user_articles.bookmarked),updated_at=now() RETURNING is_read "isRead",bookmarked,enrolled_at "enrolledAt"`,[req.user!.id,req.params.slug,body.isRead,body.bookmarked])).rows[0];
    res.json(row);
  });
  app.post('/api/articles/:slug/enroll',async(req,res)=>{
    const client=await pool.connect();
    try { await client.query('BEGIN');
      requireValue((await client.query('SELECT 1 FROM articles WHERE slug=$1',[req.params.slug])).rowCount,404,'Article not found.');
      await client.query(`INSERT INTO user_articles(user_id,article_slug,enrolled_at) VALUES($1,$2,now()) ON CONFLICT(user_id,article_slug) DO UPDATE SET enrolled_at=COALESCE(user_articles.enrolled_at,now()),updated_at=now()`,[req.user!.id,req.params.slug]);
      await client.query(`INSERT INTO user_cards(user_id,card_id,article_slug,state) SELECT $1,id,article_slug,$3::jsonb FROM cards WHERE article_slug=$2 AND NOT retired ON CONFLICT DO NOTHING`,[req.user!.id,req.params.slug,JSON.stringify(initialState())]);
      await client.query('COMMIT'); res.json({ok:true});
    } catch(error) { await client.query('ROLLBACK'); throw error; } finally {client.release();}
  });
  app.get('/api/review',async(req,res)=>{
    const params=z.object({deck:z.string().optional(),newLimit:z.coerce.number().int().min(0).max(50).default(10)}).parse(req.query);
    const rows=(await pool.query(`WITH eligible AS (SELECT c.id,c.front,c.back,c.article_slug "articleSlug",a.content->>'title' "articleTitle",uc.state,uc.due_at FROM user_cards uc JOIN cards c ON c.id=uc.card_id JOIN articles a ON a.slug=c.article_slug WHERE uc.user_id=$1 AND NOT c.retired AND ($2::text IS NULL OR c.article_slug=$2)) SELECT * FROM (SELECT * FROM eligible WHERE (state->>'attempts')::int>0 AND due_at<=now() ORDER BY due_at LIMIT 100) due UNION ALL SELECT * FROM (SELECT * FROM eligible WHERE (state->>'attempts')::int=0 ORDER BY "articleSlug",id LIMIT $3) fresh`,[req.user!.id,params.deck??null,params.newLimit])).rows;
    res.json({cards:rows.map(row=>({...row,kind:'scheduled',previews:[0,1,2,3,4,5].map(q=>schedule(row.state,q).interval)}))});
  });
  app.post('/api/reviews',async(req,res)=>{
    const payload=z.object({cardId:z.number().int().positive(),quality:z.number().int().min(0).max(5),version:z.number().int().min(0),requestId:z.uuid(),responseMs:z.number().int().min(0).max(86400000),kind:z.enum(['scheduled','practice'])}).strict().parse(req.body);
    const client=await pool.connect();
    try {
      await client.query('BEGIN');
      await client.query('SELECT id FROM users WHERE id=$1 FOR UPDATE',[req.user!.id]);
      const existing=(await client.query('SELECT payload,result FROM reviews WHERE user_id=$1 AND request_id=$2',[req.user!.id,payload.requestId])).rows[0];
      if(existing){ const same=Object.keys(payload).every(key=>existing.payload[key]===payload[key as keyof typeof payload]); requireValue(same,409,'This request ID was already used for a different review.'); await client.query('COMMIT'); res.json({...existing.result,replayed:true}); return; }
      const row=(await client.query('SELECT state,article_slug,due_at FROM user_cards WHERE user_id=$1 AND card_id=$2 FOR UPDATE',[req.user!.id,payload.cardId])).rows[0];
      requireValue(row,404,'Enroll in this deck before reviewing.');
      const state:Schedule=row.state;
      requireValue(state.version===payload.version,409,'This card changed on another device. Reload the review queue.');
      requireValue(payload.kind==='practice'?state.attempts>0:state.attempts===0||new Date(row.due_at)<=new Date(),409,'This card is not eligible for that review.');
      const nextState=schedule(state,payload.quality,new Date(),payload.kind==='practice');
      await client.query('UPDATE user_cards SET state=$3,due_at=$4 WHERE user_id=$1 AND card_id=$2',[req.user!.id,payload.cardId,JSON.stringify(nextState),nextState.due]);
      const result={state:nextState};
      await client.query('INSERT INTO reviews(user_id,card_id,request_id,quality,kind,response_ms,payload,result) VALUES($1,$2,$3,$4,$5,$6,$7,$8)',[req.user!.id,payload.cardId,payload.requestId,payload.quality,payload.kind,payload.responseMs,JSON.stringify(payload),JSON.stringify(result)]);
      await client.query('UPDATE user_articles SET updated_at=now() WHERE user_id=$1 AND article_slug=$2',[req.user!.id,row.article_slug]);
      await client.query('COMMIT'); res.json(result);
    }catch(error){await client.query('ROLLBACK');throw error;}finally{client.release();}
  });
  app.get('/api/stats',async(req,res)=>{
    const uid=req.user!.id;
    const [totals,activity,recent,forecast,learning]=await Promise.all([
      pool.query(`SELECT count(*)::int "enrolledCards",COALESCE(sum((state->>'attempts')::int),0)::int attempts,COALESCE(sum((state->>'correct')::int),0)::int correct,count(*) FILTER(WHERE (state->>'attempts')::int>0 AND due_at<=now())::int due,count(*) FILTER(WHERE (state->>'attempts')::int=0)::int fresh,count(*) FILTER(WHERE (state->>'repetitions')::int>=3 AND (state->>'interval')::int>=21)::int established FROM user_cards WHERE user_id=$1 AND card_id NOT IN (SELECT id FROM cards WHERE retired)`,[uid]),
      pool.query(`SELECT (created_at AT TIME ZONE 'UTC')::date::text "day",count(*)::int reviews FROM reviews WHERE user_id=$1 AND created_at>now()-interval '90 days' GROUP BY 1 ORDER BY 1`,[uid]),
      pool.query(`SELECT r.quality,r.kind,r.created_at "createdAt",c.front,a.slug,a.content->>'title' title FROM reviews r JOIN cards c ON c.id=r.card_id JOIN articles a ON a.slug=c.article_slug WHERE user_id=$1 ORDER BY r.created_at DESC LIMIT 30`,[uid]),
      pool.query(`SELECT (due_at AT TIME ZONE 'UTC')::date::text "day",count(*)::int cards FROM user_cards WHERE user_id=$1 AND card_id NOT IN (SELECT id FROM cards WHERE retired) AND (state->>'attempts')::int>0 AND due_at>now() AND due_at<now()+interval '14 days' GROUP BY 1 ORDER BY 1`,[uid]),
      pool.query(`SELECT count(*) FILTER(WHERE is_read)::int read,count(*) FILTER(WHERE enrolled_at IS NOT NULL)::int enrolled,(SELECT COALESCE(sum(response_ms),0)::float/60000 FROM reviews WHERE user_id=$1) "reviewMinutes" FROM user_articles WHERE user_id=$1`,[uid])
    ]);
    res.json({...totals.rows[0],...learning.rows[0],activity:activity.rows,recent:recent.rows,forecast:forecast.rows});
  });
  app.use('/api',(_req,_res,next)=>next(new HttpError(404,'API endpoint not found.')));
  const dist=fileURLToPath(new URL('../dist/',import.meta.url));
  app.use(express.static(dist,{index:false}));
  app.get(/.*/,(_req,res)=>res.sendFile(path.join(dist,'index.html')));
  app.use((error: unknown,_req:Request,res:Response,_next:NextFunction)=>{
    if(error instanceof z.ZodError) {res.status(400).json({error:'Invalid request. Check the submitted values.'});return;}
    const status=error instanceof HttpError?error.status:((error as {status?:number})?.status===400?400:500);
    if(status===500)console.error(error);
    res.status(status).json({error:status===500?'The server could not complete this request. Please retry.':error instanceof Error?error.message:'Invalid request.'});
  });
  return app;
}
