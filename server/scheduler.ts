export type Schedule = { ease: number; repetitions: number; interval: number; due: string; attempts: number; correct: number; lapses: number; version: number; practice: number };
export const initialState = (): Schedule => ({ ease: 2.5, repetitions: 0, interval: 0, due: new Date().toISOString(), attempts: 0, correct: 0, lapses: 0, version: 0, practice: 0 });
export function schedule(state: Schedule, quality: number, now = new Date(), practice = false): Schedule {
  if (!Number.isInteger(quality) || quality < 0 || quality > 5) throw new Error('Grade must be 0–5');
  if (practice) return { ...state, practice: state.practice + 1, version: state.version + 1 };
  const success = quality >= 3;
  const interval = success ? (state.repetitions === 0 ? 1 : state.repetitions === 1 ? 6 : Math.min(36500, Math.ceil(state.interval * state.ease))) : 1;
  return { ...state, ease: Math.round(Math.max(1.3, state.ease + 0.1 - (5-quality)*(0.08+(5-quality)*0.02))*100)/100, interval, repetitions: success ? state.repetitions + 1 : 0, attempts: state.attempts + 1, correct: state.correct + Number(success), lapses: state.lapses + Number(!success && state.attempts > 0), due: new Date(now.getTime()+interval*86400000).toISOString(), version: state.version+1 };
}
