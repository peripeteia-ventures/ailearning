import type { Article, Section } from './content.ts';

/** Split lines into blocks at blank lines, keeping fenced code blocks intact. */
function blocks(lines: string[]) {
  const out: string[] = []; let cur: string[] = []; let fence = false;
  for (const line of lines) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    if (!fence && !line.trim()) { if (cur.length) out.push(cur.join('\n')); cur = []; continue; }
    cur.push(line);
  }
  if (cur.length) out.push(cur.join('\n'));
  return out;
}

/** Split lines at headings of the given level, ignoring headings inside code fences. */
function splitHeadings(lines: string[], level: number) {
  const marker = '#'.repeat(level) + ' '; const parts: { title: string; lines: string[] }[] = []; let fence = false;
  for (const line of lines) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    if (!fence && line.startsWith(marker)) parts.push({ title: line.slice(marker.length).trim(), lines: [] });
    else parts.at(-1)?.lines.push(line);
  }
  return parts;
}

/** Top-level `- ` list items; indented or wrapped lines continue the current item. */
function listItems(lines: string[]) {
  const items: string[] = [];
  for (const line of lines) {
    if (line.startsWith('- ')) items.push(line.slice(2).trim());
    else if (line.trim() && items.length) items[items.length - 1] += '\n' + line;
  }
  return items;
}

/**
 * Parse an article written as Markdown (see server/content/AUTHORING.md):
 * a JSON front-matter block, then `# Sections`, `# Interview`, `# Pitfalls`,
 * `# Checklist`, `# Sources` and `# Flashcards` top-level parts.
 */
export function parseArticleMarkdown(text: string): Article {
  const match = text.replace(/\r\n/g, '\n').match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) throw new Error('Missing JSON front matter');
  const meta = JSON.parse(match[1]);
  const parts = Object.fromEntries(splitHeadings(match[2].split('\n'), 1).map(p => [p.title.toLowerCase(), p.lines]));
  for (const name of ['sections', 'interview', 'pitfalls', 'checklist', 'sources', 'flashcards']) if (!parts[name]) throw new Error(`${meta.slug}: missing # ${name}`);

  const sections: Section[] = splitHeadings(parts.sections, 2).map(s => {
    const m = s.title.match(/^(.*?)\s*\{#([a-z0-9-]+)\}$/);
    if (!m) throw new Error(`${meta.slug}: section "${s.title}" needs an {#id}`);
    return { id: m[2], title: m[1], paragraphs: blocks(s.lines) };
  });

  const interview = Object.fromEntries(splitHeadings(parts.interview, 2).map(p => [p.title.toLowerCase(), p.lines]));
  const sources = listItems(parts.sources).map(item => {
    const m = item.match(/^\[(.+?)\]\((https:[^)\s]+)\)\s*[—–-]+\s*([\s\S]+)$/);
    if (!m) throw new Error(`${meta.slug}: bad source "${item}"`);
    return { title: m[1], url: m[2], note: m[3].trim() };
  });
  const flashcards = splitHeadings(parts.flashcards, 2).map(card => {
    const [front, ...back] = blocks(card.lines);
    return { key: card.title, front: (front ?? '').replace(/^\*\*Q:\*\*\s*/, ''), back: back.join('\n\n') };
  });

  return {
    ...meta, sections,
    interview: { question: blocks(interview.question ?? []).join(' '), answer: blocks(interview.answer ?? []).join('\n\n'), followUps: listItems(interview['follow-ups'] ?? []) },
    pitfalls: listItems(parts.pitfalls), checklist: listItems(parts.checklist), sources, flashcards,
  };
}
