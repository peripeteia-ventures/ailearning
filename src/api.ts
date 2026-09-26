import { QueryClient } from '@tanstack/react-query';
import type { Article } from '../shared/content';
import type { Schedule } from '../server/scheduler';
import type { categories } from '../shared/catalog';
export class ApiError extends Error { constructor(public status:number,message:string){super(message);} }
export async function api<T>(path:string,body?:unknown,method=body?'POST':'GET'):Promise<T>{
  const response=await fetch(`/api${path}`,{method,credentials:'same-origin',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});
  const data=await response.json().catch(()=>({error:'The server returned an unreadable response.'}));
  if(!response.ok){if(response.status===401 && path!='/login' && path!='/me')queryClient.invalidateQueries({queryKey:['me']});throw new ApiError(response.status,data.error??'Request failed');}return data;
}
export const queryClient=new QueryClient({defaultOptions:{queries:{staleTime:30000,refetchOnWindowFocus:true,retry:1}}});
export type User={id:number;username:string};
export type CatalogArticle={slug:string;category:string;position:number;title:string;summary:string;difficulty:string;minutes:number;cardCount:number;sectionCount:number;diagramCount:number;isRead:boolean;bookmarked:boolean;enrolledAt:string|null;due:number;fresh:number;established:number};
export type Catalog={categories:typeof categories;articles:CatalogArticle[]};
export type Card={id:number;front:string;back:string;state:Schedule|null};
export type ArticleResponse={article:Article;progress:{isRead:boolean;bookmarked:boolean;enrolledAt:string|null};cards:Card[]};
export type ReviewCard=Card&{state:Schedule;articleSlug:string;articleTitle:string;kind:'scheduled'|'practice';previews:number[]};
export type Stats={enrolledCards:number;attempts:number;correct:number;due:number;fresh:number;established:number;read:number;enrolled:number;reviewMinutes:number;activity:{day:string;reviews:number}[];forecast:{day:string;cards:number}[];recent:{quality:number;kind:string;createdAt:string;front:string;slug:string;title:string}[]};
export function refreshStudy(){return Promise.all([queryClient.invalidateQueries({queryKey:['catalog']}),queryClient.invalidateQueries({queryKey:['stats']}),queryClient.invalidateQueries({queryKey:['article']})]);}
