import test from 'node:test';
import assert from 'node:assert/strict';
import {visualLessons} from '../shared/visuals/index.ts';
import {loadArticles} from '../server/content-loader.ts';

const articles=await loadArticles();

// Rewritten Markdown articles use "> 🎬" placeholder notes until their visuals are redrawn, so walkthroughs may be temporarily unplaced.
test('Placed visuals resolve at a valid paragraph',()=>{
  const ids=new Set(visualLessons.map(v=>v.id));
  assert.equal(ids.size,visualLessons.length,'Duplicate visual identity');
  for(const article of articles)for(const section of article.sections){
    for(const visual of section.visuals??[]){
      assert.ok(ids.has(visual.id),`${article.slug}/${section.id}: unknown visual ${visual.id}`);
      assert.ok(Number.isInteger(visual.afterParagraph)&&visual.afterParagraph>=0&&visual.afterParagraph<section.paragraphs.length,`${visual.id}: invalid insertion point`);
    }
  }
});

test('Every visual has distinct frames, readable geometry, and valid matrix cells',()=>{
  for(const lesson of visualLessons){
    assert.ok(lesson.title&&lesson.summary&&lesson.note,lesson.id);
    assert.ok(lesson.steps.length>=2,lesson.id);
    for(const [i,step] of lesson.steps.entries()){
      const context=`${lesson.id} / ${i+1}`;
      assert.ok(step.title&&step.description&&step.elements.length>=2,context);
      assert.equal(new Set(step.elements.map(e=>e.id)).size,step.elements.length,`${context}: duplicate elements`);
      if(i)assert.notEqual(JSON.stringify(step.elements),JSON.stringify(lesson.steps[i-1].elements),`${context}: picture must change`);
      for(const e of step.elements){
        for(const [key,value] of Object.entries(e))if(typeof value==='number')assert.ok(Number.isFinite(value),`${context}/${e.id}/${key}`);
        if(e.kind==='matrix'){
          const rows=e.values.length,cols=e.values[0]?.length;
          assert.ok(rows>0&&cols>0,context);
          assert.ok(e.values.every(r=>r.length===cols),`${context}: ragged matrix`);
          assert.ok(e.values.flat().every(v=>typeof v==='string'||Number.isFinite(v)),context);
          assert.ok(e.x>=0&&e.x+cols*(e.cellW??44)<=720,`${context}/${e.id}: matrix horizontal bounds`);
          assert.ok(e.y>=30&&e.y+rows*(e.cellH??34)<=380,`${context}/${e.id}: matrix vertical bounds`);
          for(const [r,c] of e.highlight??[])assert.ok(r>=0&&r<rows&&c>=0&&c<cols,context);
          if(e.rowLabels)assert.equal(e.rowLabels.length,rows,context);
          if(e.columnLabels)assert.equal(e.columnLabels.length,cols,context);
        }
        if(e.kind==='bar')assert.ok(e.value>=0&&e.value<=1,`${context}: bars use fractions`);
        if(e.kind==='text')assert.ok((e.size??16)>=14,`${context}: text too small`);
      }
    }
  }
});

test('Tiny attention and gradient examples preserve the taught arithmetic',()=>{
  const mixing=visualLessons.find(v=>v.id==='foundation-value-mixing')!;
  const output=mixing.steps.at(-1)!.elements.find(e=>e.id==='output');
  assert.ok(output?.kind==='matrix');
  const exps=[2,1,0].map(Math.exp),sum=exps.reduce((a,b)=>a+b),p=exps.map(x=>x/sum);
  const expected=[p[0]+2*p[2],2*p[1]+2*p[2]];
  output.values[0].forEach((v,i)=>assert.ok(Math.abs(Number(v)-expected[i])<0.0001));
  const gradient=visualLessons.find(v=>v.id==='training-logit-gradient');
  assert.ok(gradient,'Logit gradient walkthrough remains available');
  const g=gradient.steps[1].elements.find(e=>e.id==='g');
  assert.ok(g?.kind==='matrix');
  assert.ok(Math.abs(g.values[0].reduce<number>((s,x)=>s+Number(x),0))<1e-9,'Logit gradients sum to zero');
  assert.ok(Number(g.values[0][0])<0,'Subtracting the correct-class gradient must raise its logit');
});
