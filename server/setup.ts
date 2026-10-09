import { readFile } from 'node:fs/promises';
import { loadArticles } from './content-loader.ts';
import { fileURLToPath } from 'node:url';
import { pool } from './db.ts';
import { hashPassword } from './password.ts';
import { initialState } from './scheduler.ts';
import { articleOrder } from '../shared/catalog.ts';
export async function setup() {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query(await readFile(new URL('./schema.sql', import.meta.url),'utf8'));
    await client.query('INSERT INTO users(username,password_hash) VALUES($1,$2) ON CONFLICT DO NOTHING',['Admin', hashPassword('123')]);
    const articles = await loadArticles();
    let cardCount = 0;
    for (const { format, ...article } of articles) {
      if (!articleOrder.includes(article.slug) || (format === 'json' && article.sections.filter(x=>x.diagram).length < 5)) throw new Error(`Invalid article ${article.slug}`);
      await client.query('INSERT INTO articles(slug,category,position,content) VALUES($1,$2,$3,$4) ON CONFLICT(slug) DO UPDATE SET category=EXCLUDED.category, position=EXCLUDED.position, content=EXCLUDED.content',[article.slug,article.category,articleOrder.indexOf(article.slug),JSON.stringify(article)]);
      for (const [index,card] of article.flashcards.entries()) {
        await client.query('INSERT INTO cards(article_slug,content_key,position,front,back) VALUES($1,$2,$3,$4,$5) ON CONFLICT(content_key) DO UPDATE SET position=EXCLUDED.position,front=EXCLUDED.front,back=EXCLUDED.back,retired=false',[article.slug,`${article.slug}:${card.key}`,index,card.front,card.back]); cardCount++;
      }
      await client.query('UPDATE cards SET retired=NOT (content_key=ANY($2)) WHERE article_slug=$1',[article.slug,article.flashcards.map(card=>`${article.slug}:${card.key}`)]);
    }
    await client.query(`INSERT INTO user_cards(user_id,card_id,article_slug,state) SELECT ua.user_id,c.id,c.article_slug,$1::jsonb FROM user_articles ua JOIN cards c ON c.article_slug=ua.article_slug AND NOT c.retired WHERE ua.enrolled_at IS NOT NULL ON CONFLICT DO NOTHING`,[JSON.stringify(initialState())]);
    await client.query('DELETE FROM sessions WHERE expires_at < now()');
    await client.query('COMMIT');
    console.log(`Database ai / ailearn ready: ${articles.length} articles, ${cardCount} cards.`);
  } catch (error) { await client.query('ROLLBACK'); throw error; } finally { client.release(); }
}
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) { setup().then(()=>pool.end()).catch(error=>{ console.error(error); process.exitCode=1; return pool.end(); }); }
