export type Diagram = {
  kind: 'flow' | 'compare' | 'steps' | 'bars' | 'curve';
  title: string;
  caption: string;
  nodes?: { label: string; detail: string; value?: number }[];
  xLabel?: string;
  yLabel?: string;
  series?: { name: string; points: [number, number][] }[];
};
export type Section = {
  id: string;
  title: string;
  paragraphs: string[];
  bullets?: string[];
  formula?: string;
  code?: { language: string; title: string; value: string };
  diagram?: Diagram;
};
export type Article = {
  slug: string;
  title: string;
  category: string;
  summary: string;
  difficulty: 'Core' | 'Advanced' | 'Systems';
  minutes: number;
  prerequisites: string[];
  learningObjectives: string[];
  sections: Section[];
  interview: { question: string; answer: string; followUps: string[] };
  pitfalls: string[];
  checklist: string[];
  sources: { title: string; url: string; note: string }[];
  flashcards: { key: string; front: string; back: string }[];
};
