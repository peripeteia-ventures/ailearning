import type { ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import { AlertCircle, ArrowRight, Check, Copy, Layers, SlidersHorizontal, Network, Cpu, Workflow, ShieldCheck, Server, Compass } from 'lucide-react';
import { useState } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Empty as EmptyState, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty';
import { Spinner } from '@/components/ui/spinner';

export function Markdown({ children }: { children: string }) { return <ReactMarkdown components={{ a: ({ children, href }) => <a href={href} target="_blank" rel="noreferrer">{children}</a> }}>{children}</ReactMarkdown>; }
export function Loading() { return <div className="loading"><Spinner className="size-6" /><span>Loading your fieldnotes…</span></div>; }
export function ErrorBox({ error, retry }: { error: unknown; retry?: () => void }) { return <Alert variant="destructive" className="error-box"><AlertCircle size={20} /><AlertDescription>{error instanceof Error ? error.message : 'Something went wrong.'}{retry && <Button variant="link" className="h-auto p-0" onClick={retry}>Try again <ArrowRight size={15} /></Button>}</AlertDescription></Alert>; }
export function Empty({ title, children }: { title: string; children: ReactNode }) { return <EmptyState className="empty"><EmptyHeader><EmptyMedia><Layers size={30} /></EmptyMedia><EmptyTitle role="heading" aria-level={2}>{title}</EmptyTitle><EmptyDescription>{children}</EmptyDescription></EmptyHeader></EmptyState>; }
export function Icon({ name, size = 21 }: { name: string; size?: number }) { const Component = ({ layers: Layers, tune: SlidersHorizontal, network: Network, cpu: Cpu, workflow: Workflow, shield: ShieldCheck, server: Server, compass: Compass } as Record<string, typeof Layers>)[name] ?? Layers; return <Component size={size} />; }
export function CodeBlock({ title, language, value }: { title: string; language: string; value: string }) {
  const [copied, setCopied] = useState(false);
  return <div className="code-block"><div className="code-top"><span>{title} <small>{language}</small></span><Button variant="ghost" size="sm" aria-label={`Copy ${title}`} onClick={async () => { try { await navigator.clipboard.writeText(value); setCopied(true); setTimeout(() => setCopied(false), 1800); } catch { setCopied(false); } }}>{copied ? <Check size={16} /> : <Copy size={16} />}<span aria-live="polite">{copied ? 'Copied' : 'Copy'}</span></Button></div><pre><code>{value}</code></pre></div>;
}
export const percent = (a: number, b: number) => b ? Math.round(a / b * 100) : 0;
