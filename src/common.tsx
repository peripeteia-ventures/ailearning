import type { ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import { AlertCircle, ArrowRight, LoaderCircle, Check, Copy, Layers, SlidersHorizontal, Network, Cpu, Workflow, ShieldCheck, Server, Compass } from 'lucide-react';
import { useState } from 'react';
export function Markdown({children}:{children:string}){return <ReactMarkdown components={{a:({children,href})=><a href={href} target="_blank" rel="noreferrer">{children}</a>}}>{children}</ReactMarkdown>;}
export function Loading(){return <div className="loading"><LoaderCircle className="spin" size={26}/><span>Loading your fieldnotes…</span></div>;}
export function ErrorBox({error,retry}:{error:unknown;retry?:()=>void}){return <div role="alert" className="error-box"><AlertCircle size={20}/><div>{error instanceof Error?error.message:'Something went wrong.'}{retry&&<button onClick={retry} className="text-button">Try again <ArrowRight size={15}/></button>}</div></div>;}
export function Empty({title,children}:{title:string;children:ReactNode}){return <div className="empty"><Layers size={30}/><h2>{title}</h2><p>{children}</p></div>;}
export function Icon({name,size=21}:{name:string;size?:number}){const Component=({layers:Layers,tune:SlidersHorizontal,network:Network,cpu:Cpu,workflow:Workflow,shield:ShieldCheck,server:Server,compass:Compass} as Record<string,typeof Layers>)[name]??Layers;return <Component size={size}/>;}
export function CodeBlock({title,language,value}:{title:string;language:string;value:string}){const [copied,setCopied]=useState(false);return <div className="code-block"><div className="code-top"><span>{title} <small>{language}</small></span><button aria-label={`Copy ${title}`} onClick={async()=>{try{await navigator.clipboard.writeText(value);setCopied(true);setTimeout(()=>setCopied(false),1800);}catch{setCopied(false);}}}>{copied?<Check size={16}/>:<Copy size={16}/>} {copied?'Copied':'Copy'}</button></div><pre><code>{value}</code></pre></div>;}
export const percent=(a:number,b:number)=>b?Math.round(a/b*100):0;
