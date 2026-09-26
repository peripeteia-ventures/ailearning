import { useRef, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Link, Outlet, useRouterState } from '@tanstack/react-router';
import { BookOpen, ChartNoAxesCombined, CircleHelp, FlaskConical, LogOut, Menu, RotateCcw, ArrowUpRight, X } from 'lucide-react';
import { api, ApiError, queryClient, type User, type Stats } from './api';
import { categories } from '../shared/catalog';
import { ErrorBox, Loading } from './common';
import { useTheme } from './ThemeContext';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Sidebar, SidebarContent, SidebarFooter, SidebarHeader, SidebarInset, SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarProvider, useSidebar } from '@/components/ui/sidebar';

export function Shell() {
  const me = useQuery({ queryKey: ['me'], queryFn: () => api<User>('/me'), retry: false });
  if (me.isPending) return <Loading />;
  if (me.error) {
    if (me.error instanceof ApiError && me.error.status === 401) return <Login />;
    return <div className="standalone"><ErrorBox error={me.error} retry={() => me.refetch()} /></div>;
  }
  return <SidebarProvider open className="app-shell"><Workspace user={me.data} /></SidebarProvider>;
}

function Workspace({ user }: { user: User }) {
  const { theme } = useTheme();
  const menuTrigger = useRef<HTMLButtonElement>(null);
  const { openMobile, setOpenMobile, isMobile } = useSidebar();
  const location = useRouterState({ select: s => s.location.pathname });
  const logout = useMutation({ mutationFn: () => api('/logout', {}), onSuccess: () => { queryClient.clear(); window.location.reload(); } });
  const closeNavigation = () => setOpenMobile(false);
  return <>
    <a className="skip-link" href="#main">Skip to content</a>
    <Sidebar mobileTriggerRef={menuTrigger} className="fieldnotes-sidebar" aria-label="Workspace navigation">
      <SidebarHeader className="gap-0 p-0">
        <div className="flex items-center justify-between"><Link to="/" className="brand" onClick={closeNavigation}><Logo />latent<span>fieldnotes</span></Link>{isMobile && <Button variant="ghost" size="icon" aria-label="Close navigation" onClick={closeNavigation}><X /></Button>}</div>
        <div className="workspace-label"><span className="status-dot" />PERSONAL WORKSPACE</div>
      </SidebarHeader>
      <SidebarContent className="gap-0">
        <nav aria-label="Main navigation" onClick={closeNavigation}><SidebarMenu>
          <SidebarMenuItem><SidebarMenuButton asChild isActive={location === '/' || location.startsWith('/category') || location.startsWith('/lesson')} className="nav-item"><Link to="/"><BookOpen />Learning path</Link></SidebarMenuButton></SidebarMenuItem>
          <SidebarMenuItem><SidebarMenuButton asChild isActive={location === '/review'} className="nav-item"><Link to="/review" search={{ deck: undefined }}><RotateCcw />Spaced repetition<DueCount /></Link></SidebarMenuButton></SidebarMenuItem>
          <SidebarMenuItem><SidebarMenuButton asChild isActive={location === '/progress'} className="nav-item"><Link to="/progress"><ChartNoAxesCombined />Your progress</Link></SidebarMenuButton></SidebarMenuItem>
          <SidebarMenuItem><SidebarMenuButton asChild isActive={location === '/lab'} className="nav-item"><Link to="/lab"><FlaskConical />Concept lab</Link></SidebarMenuButton></SidebarMenuItem>
        </SidebarMenu><div className="nav-label">EXPLORE THE FIELD</div><SidebarMenu>{categories.map(c => <SidebarMenuItem key={c.slug}><SidebarMenuButton asChild isActive={location === `/category/${c.slug}`} className="topic-link"><Link to="/category/$slug" params={{ slug: c.slug }}><span style={{ background: theme.colors[`category-${c.slug}`] }} />{c.title}</Link></SidebarMenuButton></SidebarMenuItem>)}</SidebarMenu></nav>
      </SidebarContent>
      <SidebarFooter className="sidebar-bottom gap-0 p-0">
        <SidebarMenu><SidebarMenuItem><SidebarMenuButton asChild isActive={location === '/guide'} className="nav-item"><Link to="/guide" onClick={closeNavigation}><CircleHelp />How to study here<ArrowUpRight className="ml-auto" /></Link></SidebarMenuButton></SidebarMenuItem></SidebarMenu>
        {logout.error && <ErrorBox error={logout.error} />}
        <div className="user-row"><Avatar><AvatarFallback>{user.username[0]?.toUpperCase()}</AvatarFallback></Avatar><div className="min-w-0"><strong className="truncate">{user.username}</strong><small>Keep your curiosity.</small></div><Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="ml-auto" aria-label="Sign out" disabled={logout.isPending} onClick={() => logout.mutate()}><LogOut size={17} /></Button></TooltipTrigger><TooltipContent>Sign out</TooltipContent></Tooltip></div>
      </SidebarFooter>
    </Sidebar>
    <SidebarInset className="main site-main">
      <div className="mobile-header"><Link to="/" className="brand"><Logo />latent<span>fieldnotes</span></Link><Button variant="ghost" size="icon" ref={menuTrigger} aria-label="Open navigation" aria-expanded={openMobile} onClick={() => setOpenMobile(true)}><Menu /></Button></div>
      <div className="topbar"><span>AI SYSTEMS / <strong>{location === '/review' ? 'RECALL' : location === '/progress' ? 'PROGRESS' : location === '/lab' ? 'EXPERIMENTS' : 'THE FIELD GUIDE'}</strong></span><span className="topbar-note"><span className="status-dot" />Built for deeper understanding</span></div>
      <div id="main" tabIndex={-1}><Outlet /></div>
    </SidebarInset>
  </>;
}

function DueCount() { const stats = useQuery({ queryKey: ['stats'], queryFn: () => api<Stats>('/stats') }); return stats.data?.due ? <Badge className="nav-count">{stats.data.due}</Badge> : null; }
export function Logo() { return <span className="logo"><span /><i /></span>; }
function Login() {
  const [username, setUsername] = useState('Admin'); const [password, setPassword] = useState('');
  const login = useMutation({ mutationFn: () => api<User>('/login', { username, password }), onSuccess: user => { queryClient.setQueryData(['me'], user); } });
  return <div className="login-page"><div className="login-art"><div className="brand"><Logo />latent<span>fieldnotes</span></div><div><span className="eyebrow">UNDERSTAND THE SYSTEM. NOT JUST THE ANSWER.</span><h1>Small notes.<br />Deep understanding.</h1><p>A field guide to the ideas behind modern AI.<br />From the first gradient to the last token.</p><div className="login-network" aria-hidden="true">{[0, 1, 2, 3, 4].map(i => <div key={i} style={{ transform: `translateY(${Math.sin(i) * 25}px)` }}>{[0, 1, 2].map(j => <i key={j} style={{ opacity: .35 + (i + j) % 3 * .2 }} />)}</div>)}</div></div><span className="login-footer">ARCHITECTURE · THEORY · SYSTEMS · PRACTICE</span></div><div className="login-form"><span className="eyebrow">YOUR PERSONAL LEARNING SPACE</span><h2>Welcome back.</h2><p>Pick up a concept. Make it stick.</p><form onSubmit={e => { e.preventDefault(); login.mutate(); }}>
    <div className="grid gap-2.5"><Label htmlFor="username">Username</Label><Input id="username" autoComplete="username" value={username} onChange={e => setUsername(e.target.value)} required /></div>
    <div className="grid gap-2.5"><Label htmlFor="password">Password</Label><Input id="password" type="password" autoComplete="current-password" placeholder="Enter your password" value={password} onChange={e => setPassword(e.target.value)} required /></div>
    {login.error && <ErrorBox error={login.error} />}<Button type="submit" size="lg" disabled={login.isPending}>{login.isPending ? 'Signing in…' : 'Open your fieldnotes'}<ArrowUpRight size={18} /></Button>
  </form><p className="login-caption">Read carefully. Recall honestly. Connect the ideas.</p></div></div>;
}
