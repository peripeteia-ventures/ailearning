import type { Diagram as DiagramData } from '../shared/content';
import { ArrowRight } from 'lucide-react';
const colors=['#c5ef82','#94c9ea','#d0aceb','#efbd84'];
export function Diagram({diagram:d,index}:{diagram:DiagramData;index:number}){
  const all=d.series?.flatMap(s=>s.points)??[];
  const minX=Math.min(...all.map(p=>p[0])),maxX=Math.max(...all.map(p=>p[0]));
  const minY=Math.min(0,...all.map(p=>p[1])),maxY=Math.max(...all.map(p=>p[1]));
  const x=(n:number)=>65+(n-minX)/(maxX-minX||1)*570;
  const y=(n:number)=>245-(n-minY)/(maxY-minY||1)*205;
  return <figure className={`diagram diagram-${d.kind}`}><div className="diagram-heading"><span className="eyebrow">FIG {String(index).padStart(2,'0')}</span><strong>{d.title}</strong><span className="diagram-dot"/></div>
    {d.kind==='curve'?<><svg viewBox="0 0 700 300" role="img" aria-label={`${d.title}. ${d.caption}`}>
      {[0,.25,.5,.75,1].map(t=><g key={t}><line x1="65" x2="635" y1={245-t*205} y2={245-t*205} stroke="#334135" strokeDasharray="3 6"/><text x="55" y={250-t*205} textAnchor="end">{Number((minY+t*(maxY-minY)).toPrecision(3))}</text></g>)}
      <line x1="65" x2="635" y1="245" y2="245" stroke="#6a7769"/>
      {[0,.25,.5,.75,1].map(t=><text key={t} x={65+t*570} y="265" textAnchor="middle">{Number((minX+t*(maxX-minX)).toPrecision(3))}</text>)}
      {d.series?.map((s,i)=><g key={s.name}><polyline points={s.points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')} fill="none" stroke={colors[i%4]} strokeWidth="3" strokeLinejoin="round"/>{s.points.length<30&&s.points.map((p,j)=><circle key={j} cx={x(p[0])} cy={y(p[1])} r="3" fill={colors[i%4]}/>)}</g>)}
      <text x="350" y="292" textAnchor="middle">{d.xLabel}</text><text x="65" y="19">{d.yLabel}</text>
    </svg><div className="legend">{d.series?.map((s,i)=><span key={s.name}><i style={{background:colors[i%4]}}/>{s.name}</span>)}</div><details className="chart-data"><summary>View plot values</summary>{d.series?.map(s=><p key={s.name}><b>{s.name}:</b> {s.points.map(p=>`(${p[0]}, ${p[1]})`).join(' · ')}</p>)}</details></>:
    d.kind==='bars'?<div className="diagram-bars">{d.nodes?.map((n,i)=><div key={i}><div className="bar-label"><strong>{n.label}</strong><span>{n.value}</span></div><div className="bar-track"><div style={{width:`${Math.max(0,(n.value??0)/Math.max(1,...(d.nodes??[]).map(x=>x.value??0))*100)}%`,background:colors[i%4]}}/></div><p>{n.detail}</p></div>)}</div>:
    <div className={`diagram-nodes ${d.kind==='compare'?'comparison':''}`}>{d.nodes?.map((n,i)=><div className="diagram-node" key={i}><span className="node-number">{String(i+1).padStart(2,'0')}</span><strong>{n.label}</strong><MarkdownText text={n.detail}/>{d.kind!=='compare'&&i<(d.nodes?.length??0)-1&&<ArrowRight className="node-arrow" size={16}/>}</div>)}</div>}
    <figcaption>{d.caption}</figcaption></figure>;
}
function MarkdownText({text}:{text:string}){return <p>{text}</p>;}
