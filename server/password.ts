import { randomBytes, scryptSync, timingSafeEqual, createHash } from 'node:crypto';
export function hashPassword(password: string) { const salt = randomBytes(16).toString('hex'); return `${salt}:${scryptSync(password, salt, 64).toString('hex')}`; }
export function verifyPassword(password: string, hash: string) { const [salt,key] = hash.split(':'); if (!salt || !key) return false; const actual = scryptSync(password, salt, 64); const expected = Buffer.from(key, 'hex'); return actual.length === expected.length && timingSafeEqual(actual, expected); }
export function tokenHash(token: string) { return createHash('sha256').update(token).digest('hex'); }
