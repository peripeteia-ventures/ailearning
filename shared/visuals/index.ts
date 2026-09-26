import { foundationVisuals } from './foundations.ts';
import { architectureVisuals } from './architecture.ts';
import { trainingVisuals } from './training.ts';
import { applicationVisuals } from './applications.ts';
import type { VisualLesson } from './types.ts';

export const visualLessons: VisualLesson[] = [...foundationVisuals, ...architectureVisuals, ...trainingVisuals, ...applicationVisuals];
