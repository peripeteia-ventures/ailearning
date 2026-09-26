import { useEffect, useId, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from 'lucide-react';
import { visualLessons } from '../shared/visuals/index';
import type { Tone, VisualElement, VisualLesson as Lesson } from '../shared/visuals/types';
import './visual-lessons.css';

const palette: Record<Tone, string> = { green: 'var(--chart-primary)', blue: 'var(--chart-blue)', purple: 'var(--chart-purple)', amber: 'var(--chart-amber)', red: 'var(--chart-negative)', muted: 'var(--muted)' };
const tones = Object.keys(palette) as Tone[];

function Matrix({ e }: { e: Extract<VisualElement, {kind:'matrix'}> }) {
  const w = e.cellW ?? 44, h = e.cellH ?? 34;
  return <g className="vl-matrix" transform={`translate(${e.x} ${e.y})`}>
    {e.label && <text x="0" y={e.columnLabels ? -34 : -14} className="vl-label">{e.label}</text>}
    {e.columnLabels?.map((label, i) => <text key={i} x={i*w+w/2} y="-12" textAnchor="middle" className="vl-axis-label">{label}</text>)}
    {e.rowLabels?.map((label, i) => <text key={i} x="-10" y={i*h+h/2+5} textAnchor="end" className="vl-axis-label">{label}</text>)}
    {e.values.map((row,r) => row.map((value,c) => {
      const selected = e.highlight?.some(([i,j]) => i===r && j===c);
      const negative = typeof value === 'number' && value < 0;
      return <g key={`${r}-${c}`}><rect x={c*w+2} y={r*h+2} width={w-4} height={h-4} rx="4" fill={selected?'currentColor':'var(--surface)'} fillOpacity={selected ? .23 : 1} stroke="currentColor" strokeOpacity={selected?1:.35} strokeWidth={selected?2:1}/><text x={c*w+w/2} y={r*h+h/2+5} textAnchor="middle" className="vl-number" style={{fill:negative?'var(--chart-negative)':selected?'currentColor':'var(--text)'}}>{value}</text></g>;
    }))}
  </g>;
}

function Element({ e, prefix }: { e: VisualElement; prefix: string }) {
  let node;
  switch(e.kind) {
    case 'text': node = <text x={e.x} y={e.y} fontSize={e.size??16} textAnchor={e.anchor??'start'}>{e.text}</text>; break;
    case 'matrix': node = <Matrix e={e}/>; break;
    case 'box': node = <g><rect x={e.x} y={e.y} width={e.w} height={e.h} rx="9" stroke="currentColor" strokeOpacity={e.active?1:.5} strokeWidth={e.active?2:1} fill="currentColor" fillOpacity={e.active ? .15 : .045}/><text x={e.x+e.w/2} y={e.y+e.h/2+(e.detail?-4:5)} textAnchor="middle" className="vl-box-label">{e.label}</text>{e.detail&&<text x={e.x+e.w/2} y={e.y+e.h/2+19} textAnchor="middle" className="vl-box-detail">{e.detail}</text>}</g>; break;
    case 'arrow': {
      const mx=(e.x1+e.x2)/2,my=(e.y1+e.y2)/2;
      const path=e.bend?`M${e.x1},${e.y1} Q${mx},${my+e.bend} ${e.x2},${e.y2}`:`M${e.x1},${e.y1} L${e.x2},${e.y2}`;
      node=<g><path d={path} fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray={e.dashed?'5 5':undefined} markerEnd={`url(#${prefix}-${e.tone??'green'})`}/>{e.label&&<text x={mx} y={my+(e.bend??0)/2-10} textAnchor="middle" className="vl-arrow-label">{e.label}</text>}</g>; break;
    }
    case 'path': node=<path d={e.d} stroke="currentColor" strokeWidth={e.width??2} fill={e.fill?'currentColor':'none'} fillOpacity={e.fill ? .1 : undefined} strokeDasharray={e.dashed?'5 5':undefined}/>; break;
    case 'circle': node=<g><circle cx={e.x} cy={e.y} r={e.r} stroke="currentColor" strokeWidth="2" fill={e.filled?'currentColor':'var(--bg)'}/>{e.label&&<text x={e.x} y={e.y+5} textAnchor="middle" style={{fill:e.filled?'var(--bg)':'currentColor'}}>{e.label}</text>}</g>; break;
    case 'bar': node=<g><rect x={e.x} y={e.y} width={e.w} height={e.h} rx="4" fill="var(--surface2)"/><rect x={e.x} y={e.y} width={e.w*Math.max(0,Math.min(1,e.value))} height={e.h} rx="4" fill="currentColor"/>{e.label&&<text x={e.x} y={e.y-9} className="vl-label">{e.label}</text>}</g>; break;
  }
  return <g data-element={e.id} style={{color:palette[e.tone??'green'],opacity:e.opacity??1}}>{node}</g>;
}

function describeElements(elements: VisualElement[]) {
  return elements.filter(e=>(e.opacity??1)>0).map(e=>{
    switch(e.kind) {
      case 'text': return e.text;
      case 'matrix': return `${e.label??'Matrix'}: ${e.values.map((row,i)=>`${e.rowLabels?.[i]??`row ${i+1}`}: ${row.join(', ')}`).join('; ')}`;
      case 'box': return `${e.label}${e.detail?`: ${e.detail}`:''}`;
      case 'bar': return `${e.label??'Bar'}: ${Math.round(e.value*100)}%`;
      default: return 'label' in e?e.label:'';
    }
  }).filter(Boolean).join('. ');
}

export function VisualLesson({ id }: { id: string }) {
  const lesson = visualLessons.find(v=>v.id===id);
  return lesson ? <Player key={id} lesson={lesson}/> : null;
}

function Player({lesson}:{lesson:Lesson}) {
  const [step,setStep]=useState(0),[playing,setPlaying]=useState(false),[slow,setSlow]=useState(false);
  const [reduced,setReduced]=useState(false);
  const figure=useRef<HTMLElement>(null), prefix=useId().replace(/:/g,'');
  const current=lesson.steps[step], last=lesson.steps.length-1;
  useEffect(()=>{
    const query=matchMedia('(prefers-reduced-motion: reduce)');
    const sync=()=>{setReduced(query.matches);if(query.matches)setPlaying(false);};
    sync();query.addEventListener('change',sync);return()=>query.removeEventListener('change',sync);
  },[]);
  useEffect(()=>{
    const pause=()=>{if(document.hidden)setPlaying(false);};
    const other=(event:Event)=>{if((event as CustomEvent).detail!==prefix)setPlaying(false);};
    const observer=new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)setPlaying(false);});
    if(figure.current)observer.observe(figure.current);
    document.addEventListener('visibilitychange',pause);window.addEventListener('visual-lesson-play',other);
    return()=>{observer.disconnect();document.removeEventListener('visibilitychange',pause);window.removeEventListener('visual-lesson-play',other);};
  },[prefix]);
  useEffect(()=>{
    if(!playing)return;
    if(step===last){setPlaying(false);return;}
    const timer=window.setTimeout(()=>setStep(s=>s+1),slow?6000:3500);
    return()=>window.clearTimeout(timer);
  },[playing,step,last,slow]);
  const select=(n:number)=>{setPlaying(false);setStep(Math.max(0,Math.min(last,n)));};
  const play=()=>{
    if(playing){setPlaying(false);return;}
    if(step===last)setStep(0);
    window.dispatchEvent(new CustomEvent('visual-lesson-play',{detail:prefix}));setPlaying(true);
  };
  return <figure ref={figure} className="visual-lesson" data-visual-id={lesson.id} aria-labelledby={`${prefix}-title`}>
    <figcaption className="vl-heading"><span className="eyebrow">SEE THE STRUCTURE</span><h3 id={`${prefix}-title`}>{lesson.title}</h3><p>{lesson.summary}</p></figcaption>
    <div className="vl-canvas" tabIndex={0} role="region" aria-label={`${lesson.title} diagram. Scroll horizontally on small screens.`}>
      <svg viewBox="0 0 720 380" role="img" aria-labelledby={`${prefix}-svg-title ${prefix}-svg-desc`}>
        <title id={`${prefix}-svg-title`}>{current.title}</title><desc id={`${prefix}-svg-desc`}>{current.description} {describeElements(current.elements)}</desc>
        <defs>{tones.map(t=><marker key={t} id={`${prefix}-${t}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" fill={palette[t]}/></marker>)}</defs>
        {current.elements.map(e=><Element e={e} prefix={prefix} key={`${e.id}-${e.kind}`}/>)}
      </svg>
    </div>
    <p className="vl-pan-hint">Swipe the picture sideways to see every part.</p>
    <div className="vl-narration" aria-live="polite" aria-atomic="true"><span className="vl-step-count">{String(step+1).padStart(2,'0')} / {String(lesson.steps.length).padStart(2,'0')}</span><div><h4>{current.title}</h4><p>{current.description}</p></div></div>
    <div className="vl-controls">
      <div className="vl-playback"><button type="button" className="vl-play" onClick={play} disabled={reduced} aria-label={playing?'Pause animation':step===last?'Replay animation':'Play animation'}>{playing?<Pause size={16}/>:<Play size={16}/>}<span>{playing?'Pause':step===last?'Replay':'Play'}</span></button><button type="button" onClick={()=>select(0)} aria-label="Reset animation" disabled={step===0&&!playing}><RotateCcw size={16}/></button><label className="vl-speed"><input type="checkbox" checked={slow} onChange={e=>setSlow(e.target.checked)}/> Slower</label></div>
      <div className="vl-stepping"><button type="button" onClick={()=>select(step-1)} disabled={step===0} aria-label="Previous step"><ChevronLeft size={17}/></button><span>Step {step+1} of {lesson.steps.length}</span><button type="button" onClick={()=>select(step+1)} disabled={step===last} aria-label="Next step"><ChevronRight size={17}/></button></div>
    </div>
    <ol className="vl-step-list" aria-label="Choose an animation step">{lesson.steps.map((s,i)=><li key={i}><button type="button" aria-current={i===step?'step':undefined} onClick={()=>select(i)}><span>{i+1}</span>{s.title}</button></li>)}</ol>
    {reduced&&<p className="vl-motion-note">Reduced motion is on. Use the step buttons to explore each picture.</p>}
    <p className="vl-note">{lesson.note}</p>
    <details className="vl-transcript"><summary>Read the visual walkthrough</summary><ol>{lesson.steps.map((s,i)=><li key={i}><strong>{s.title}</strong><p>{s.description}</p></li>)}</ol><p><strong>Current picture values:</strong> {describeElements(current.elements)}</p></details>
  </figure>;
}
