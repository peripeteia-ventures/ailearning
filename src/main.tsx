import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClientProvider } from '@tanstack/react-query';
import { createRootRoute, createRoute, createRouter, RouterProvider, lazyRouteComponent } from '@tanstack/react-router';
import { queryClient } from './api';
import { Shell } from './Shell';
import { Learn } from './Learn';
import './styles.css';
const root=createRootRoute({component:Shell,notFoundComponent:()=> <div className="empty"><h1>That page wandered off.</h1><a href="/">Return to the learning path</a></div>});
const index=createRoute({getParentRoute:()=>root,path:'/',component:Learn});
const category=createRoute({getParentRoute:()=>root,path:'/category/$slug',component:Learn});
const article=createRoute({getParentRoute:()=>root,path:'/lesson/$slug',component:lazyRouteComponent(()=>import('./ArticlePage'),'ArticlePage')});
const review=createRoute({getParentRoute:()=>root,path:'/review',component:lazyRouteComponent(()=>import('./Review'),'Review'),validateSearch:(s:Record<string,unknown>)=>({deck:typeof s.deck==='string'?s.deck:undefined})});
const progress=createRoute({getParentRoute:()=>root,path:'/progress',component:lazyRouteComponent(()=>import('./Progress'),'Progress')});
const guide=createRoute({getParentRoute:()=>root,path:'/guide',component:lazyRouteComponent(()=>import('./Progress'),'Guide')});
const lab=createRoute({getParentRoute:()=>root,path:'/lab',component:lazyRouteComponent(()=>import('./Lab'),'Lab')});
const router=createRouter({routeTree:root.addChildren([index,category,article,review,progress,guide,lab]),scrollRestoration:true});
declare module '@tanstack/react-router' { interface Register {router:typeof router} }
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><QueryClientProvider client={queryClient}><RouterProvider router={router}/></QueryClientProvider></React.StrictMode>);
