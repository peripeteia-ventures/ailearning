// Dependency-free framing tests only; these do not replace database integration tests.
import {readFile} from 'node:fs/promises';
import {runInNewContext} from 'node:vm';
import assert from 'node:assert/strict';
const source=await readFile(new URL('./server.js',import.meta.url),'utf8');
const parser=source.slice(source.indexOf('async function* sse('),source.indexOf('async function upstream('));
const sse=runInNewContext(`${parser}; sse`,{TextDecoder});
const encoder=new TextEncoder();
async function collect(bytes,chunkSize){let offset=0;const body=new ReadableStream({pull(c){if(offset>=bytes.length){c.close();return;}c.enqueue(bytes.slice(offset,offset+chunkSize));offset+=chunkSize;}});const result=[];for await(const item of sse(body))result.push(item);return result;}
const data=encoder.encode(': comment\r\nevent: test\r\ndata: {"text":"café ☕"}\r\n\r\ndata: first\r\ndata: second\r\n\r\ndata: [DONE]\r\n\r\n');
for(const size of [1,2,3,7,128])assert.deepEqual(await collect(data,size),['{"text":"café ☕"}','first\nsecond','[DONE]']);
assert.deepEqual(await collect(encoder.encode('data: incomplete'),1),[]);
await assert.rejects(()=>collect(encoder.encode('data: '+'x'.repeat(131073)),1024),/upstream_frame_limit/);
console.log('Passed fragmented UTF-8, CRLF, multiline data, EOF, and frame-size tests.');
