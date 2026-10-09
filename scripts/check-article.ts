// Usage: node scripts/check-article.ts SLUG — parses server/content/articles/SLUG.md and reports its shape.
import { readFile } from 'node:fs/promises';
import { parseArticleMarkdown } from '../shared/article-markdown.ts';
import { articleOrder, categories } from '../shared/catalog.ts';

const slug = process.argv[2];
const a = parseArticleMarkdown(await readFile(new URL(`../server/content/articles/${slug}.md`, import.meta.url), 'utf8'));
const paragraphs = a.sections.flatMap(s => s.paragraphs);
const words = paragraphs.join(' ').split(/\s+/).length;
const placeholders = paragraphs.filter(p => p.startsWith('> 🎬')).length;
const problems = [
  a.slug !== slug && 'slug does not match file name',
  !categories.some(c => c.slug === a.category) && `unknown category ${a.category}`,
  ...a.prerequisites.filter(p => !articleOrder.includes(p)).map(p => `unknown prerequisite ${p}`),
  a.sections.length < 6 && 'fewer than 6 sections',
  words < 1200 && 'fewer than 1200 words',
  placeholders < 3 && 'fewer than 3 🎬 placeholders',
  (a.flashcards.length < 12 || a.flashcards.length > 16) && 'flashcards must number 12–16',
  new Set(a.flashcards.map(c => c.key)).size !== a.flashcards.length && 'duplicate flashcard keys',
  a.flashcards.some(c => !c.front || !c.back) && 'flashcard missing front/back',
  a.sources.length < 4 && 'fewer than 4 sources',
  a.interview.followUps.length < 2 && 'fewer than 2 follow-ups',
  (a.pitfalls.length < 3 || a.checklist.length < 3) && 'needs 3+ pitfalls and checklist items',
  a.learningObjectives.length < 3 && 'needs 3+ learning objectives',
  paragraphs.some(p => /^```\s*$/m.test(p.split('\n')[0])) && 'code fence without a language',
].filter(Boolean);
console.log(`${a.slug}: ${a.sections.length} sections, ${words} words, ${placeholders} placeholders, ${a.flashcards.length} cards, ${a.sources.length} sources`);
for (const s of a.sections) console.log(`  - ${s.title} {#${s.id}}`);
if (problems.length) { console.error('PROBLEMS:\n  ' + problems.join('\n  ')); process.exitCode = 1; } else console.log('OK');
