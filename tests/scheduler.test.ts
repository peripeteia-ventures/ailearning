import test from 'node:test';
import assert from 'node:assert/strict';
import { initialState,schedule } from '../server/scheduler.ts';
test('SM-2 success intervals use prior ease and elapsed UTC days',()=>{const now=new Date('2026-03-08T08:00:00Z');const one=schedule(initialState(),4,now);assert.equal(one.interval,1);assert.equal(one.due,'2026-03-09T08:00:00.000Z');const two=schedule(one,4,now);assert.equal(two.interval,6);const three=schedule(two,5,now);assert.equal(three.interval,15);assert.equal(three.ease,2.6);assert.equal(three.correct,3);});
test('Failures reset repetitions, preserve minimum ease, and count later lapses',()=>{let s=initialState();s=schedule(s,0);assert.equal(s.lapses,0);s=schedule(s,0);assert.equal(s.lapses,1);assert.equal(s.ease,1.3);assert.equal(s.interval,1);assert.equal(s.repetitions,0);assert.equal(s.correct,0);});
test('Practice increments only practice and version',()=>{const s=schedule(initialState(),3);const p=schedule(s,0,new Date(),true);assert.deepEqual({...p,practice:s.practice,version:s.version},s);assert.equal(p.practice,1);assert.equal(p.version,s.version+1);});
test('Invalid grades fail and intervals have a finite ceiling',()=>{for(const grade of [-1,6,1.5,NaN])assert.throws(()=>schedule(initialState(),grade));assert.equal(schedule({...initialState(),repetitions:10,interval:36500},5).interval,36500);});
