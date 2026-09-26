export type Tone = 'green' | 'blue' | 'purple' | 'amber' | 'red' | 'muted';
type Base = { id: string; tone?: Tone; opacity?: number };
export type VisualElement = Base & (
  | { kind: 'text'; x: number; y: number; text: string; size?: number; anchor?: 'start' | 'middle' | 'end' }
  | { kind: 'box'; x: number; y: number; w: number; h: number; label: string; detail?: string; active?: boolean }
  | { kind: 'matrix'; x: number; y: number; values: (number | string)[][]; label?: string; rowLabels?: string[]; columnLabels?: string[]; cellW?: number; cellH?: number; highlight?: [number, number][] }
  | { kind: 'arrow'; x1: number; y1: number; x2: number; y2: number; label?: string; bend?: number; dashed?: boolean }
  | { kind: 'path'; d: string; width?: number; dashed?: boolean; fill?: boolean }
  | { kind: 'circle'; x: number; y: number; r: number; label?: string; filled?: boolean }
  | { kind: 'bar'; x: number; y: number; w: number; h: number; value: number; label?: string }
);
export type VisualStep = { title: string; description: string; elements: VisualElement[] };
export type VisualLesson = { id: string; title: string; summary: string; note: string; steps: VisualStep[] };
