import { readFile, readdir } from 'node:fs/promises';
import { parseArticleMarkdown } from '../shared/article-markdown.ts';
import type { Article } from '../shared/content.ts';

const legacyDir = new URL('./content/', import.meta.url);
const markdownDir = new URL('./content/articles/', import.meta.url);

/** Rewritten articles live in content/articles/SLUG.md and replace a legacy content/SLUG.json with the same slug. */
export async function loadArticles(): Promise<(Article & { format: 'markdown' | 'json' })[]> {
  const bySlug = new Map<string, Article & { format: 'markdown' | 'json' }>();
  for (const file of (await readdir(legacyDir)).filter(f => f.endsWith('.json'))) {
    const article: Article = JSON.parse(await readFile(new URL(file, legacyDir), 'utf8'));
    bySlug.set(article.slug, { ...article, format: 'json' });
  }
  for (const file of (await readdir(markdownDir)).filter(f => f.endsWith('.md'))) {
    const article = parseArticleMarkdown(await readFile(new URL(file, markdownDir), 'utf8'));
    if (`${article.slug}.md` !== file) throw new Error(`${file}: slug "${article.slug}" does not match file name`);
    bySlug.set(article.slug, { ...article, format: 'markdown' });
  }
  return [...bySlug.values()];
}
