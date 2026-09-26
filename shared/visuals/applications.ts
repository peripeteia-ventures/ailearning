import type { VisualElement as E, VisualLesson } from './types.ts';

const vectorPicture = (stage:number):E[] => [
  {id:'x-axis',kind:'arrow',x1:65,y1:240,x2:320,y2:240,tone:'muted'},
  {id:'y-axis',kind:'arrow',x1:155,y1:290,x2:155,y2:65,tone:'muted'},
  {id:'x-label',kind:'text',x:305,y:268,text:'x₁',tone:'muted'},
  {id:'y-label',kind:'text',x:135,y:60,text:'x₂',tone:'muted'},
  {id:'q',kind:'arrow',x1:155,y1:240,x2:265,y2:185,tone:'green'},
  {id:'q-label',kind:'text',x:270,y:195,text:'q [2,1]',tone:'green'},
  {id:'a',kind:'arrow',x1:155,y1:240,x2:265,y2:130,tone:'blue',opacity:stage>0?1:.2},
  {id:'a-label',kind:'text',x:260,y:113,text:'A [2,2]',tone:'blue',opacity:stage>0?1:.2},
  {id:'b',kind:'arrow',x1:155,y1:240,x2:100,y2:130,tone:'purple',opacity:stage>0?1:.2},
  {id:'b-label',kind:'text',x:50,y:113,text:'B [−1,2]',tone:'purple',opacity:stage>0?1:.2},
  {id:'c',kind:'arrow',x1:155,y1:240,x2:210,y2:240,tone:'amber',opacity:stage>0?1:.2},
  {id:'c-label',kind:'text',x:185,y:285,text:'C [1,0]',tone:'amber',opacity:stage>0?1:.2},
  {id:'q-values',kind:'matrix',x:430,y:85,values:[[2,1]],label:'Query coordinates',cellW:65,cellH:42},
  ...(stage>0?[{id:'cosine',kind:'text',x:385,y:167,text:'Compare direction, not length',size:16,tone:'muted'} as E,
    ...(['A','C','B'] as const).flatMap((name,i):E[]=>[{id:`bar-${name}`,kind:'bar',x:410,y:205+i*49,w:220,h:15,value:stage>1?[.948683,.894427,0][i]:0,label:`${name} · ${stage>1?['0.949','0.894','0.000'][i]:'cosine score'}`,tone:['blue','amber','purple'][i] as 'blue'|'amber'|'purple'}])]:[]),
  {id:'origin',kind:'text',x:137,y:261,text:'0',size:14,tone:'muted'},
];

const graphPicture=(stage:number):E[]=>{
  const nodes=[[120,270],[230,270],[340,270],[450,270],[560,270]];
  return [
    {id:'top-label',kind:'text',x:55,y:53,text:'Sparse upper layer',tone:'muted'},
    {id:'base-label',kind:'text',x:55,y:221,text:'Base layer · all stored vectors',tone:'muted'},
    {id:'upper-edge',kind:'path',d:'M120,100 L450,100',tone:'muted'},
    {id:'upper-a',kind:'circle',x:120,y:100,r:22,label:'A',tone:stage===0?'green':'muted',filled:stage===0},
    {id:'upper-d',kind:'circle',x:450,y:100,r:22,label:'D',tone:'green',filled:stage>0},
    ...[[0,1],[1,2],[2,3],[3,4],[1,3]].map(([a,b],i):E=>({id:`edge-${i}`,kind:'path',d:`M${nodes[a][0]},270 Q${(nodes[a][0]+nodes[b][0])/2},${i===4?185:270} ${nodes[b][0]},270`,tone:'muted'})),
    ...nodes.map(([x,y],i):E=>({id:`node-${i}`,kind:'circle',x,y,r:22,label:['A','B','C','D','E'][i],tone:stage>=2&&i>=3?'blue':'muted',filled:stage===3&&i===4})),
    {id:'query',kind:'circle',x:595,y:152,r:20,label:'q',tone:'amber',filled:true},
    ...(stage===1?[{id:'hop',kind:'arrow',x1:147,y1:80,x2:422,y2:80,tone:'green',label:'closer to q'} as E]:[]),
    ...(stage>=2?[{id:'descend',kind:'arrow',x1:450,y1:128,x2:450,y2:238,tone:'green',dashed:true} as E]:[]),
    ...(stage===3?[{id:'explore',kind:'arrow',x1:478,y1:247,x2:533,y2:247,tone:'blue'} as E]:[]),
    {id:'reminder',kind:'text',x:55,y:345,text:stage===3?'Explore a neighborhood; a limited search can miss neighbors.':'Copies of a node across layers refer to the same stored vector.',size:16,tone:'muted'},
  ];
};

const speculativePicture=(stage:number):E[]=>[
  {id:'prefix-label',kind:'text',x:55,y:50,text:'Committed prefix',tone:'muted'},
  {id:'prefix',kind:'matrix',x:55,y:70,values:[['The','cat']],cellW:74,cellH:40,tone:'green'},
  {id:'draft-label',kind:'text',x:270,y:50,text:'Draft proposes a suffix',tone:'blue'},
  ...['sat','near','me'].map((token,i):E=>({id:`draft-${i}`,kind:'box',x:270+i*115,y:70,w:100,h:40,label:token,tone:stage>=2?(i===0?'green':'red'):'blue',opacity:stage===3&&i>0?.3:1,active:stage===2})),
  ...(stage===1?[{id:'verify-label',kind:'text',x:270,y:157,text:'Target checks the same prefixes',tone:'purple'} as E,...[0,1,2].map((i):E=>({id:`verify-${i}`,kind:'arrow',x1:320+i*115,y1:174,x2:320+i*115,y2:118,tone:'purple'}))]:[]),
  ...(stage>=2?[
    {id:'reject-title',kind:'text',x:55,y:175,text:'At “near”: q = 0.50, p = 0.20',tone:'red'} as E,
    {id:'accept-rule',kind:'text',x:55,y:207,text:'Accept with min(1, p/q) = 0.40',tone:'muted'} as E,
    {id:'random',kind:'text',x:55,y:239,text:'Draw u = 0.70 → reject this proposal',tone:'red'} as E,
  ]:[]),
  ...(stage===3?[
    {id:'correction',kind:'box',x:420,y:178,w:215,h:67,label:'Residual distribution',detail:'normalize max(p − q, 0)',tone:'amber'} as E,
    {id:'result',kind:'matrix',x:55,y:300,values:[['The','cat','sat','on']],cellW:85,cellH:40,label:'Commit accepted token + correction',tone:'green'} as E,
    {id:'discard',kind:'text',x:430,y:320,text:'Discard “near me”.',tone:'red'} as E,
  ]:[]),
];

export const applicationVisuals:VisualLesson[]=[
  {id:'retrieval-vector-geometry',title:'A vector is a direction you can compare',summary:'Give a query and three passages just two coordinates, then see what cosine similarity measures.',note:'Invented two-dimensional embeddings; coordinates have no assigned semantic meaning. Real embeddings have many dimensions, and a two-dimensional projection can distort their relationships. Cosine scores are not relevance probabilities.',steps:[
    {title:'Draw the query',description:'The query q = [2,1] moves two units along the first axis and one along the second. The two cells and the green arrow are two views of the same vector.',elements:vectorPicture(0)},
    {title:'Add passage vectors',description:'Each passage has its own direction. Passage A points close to the query; C points along the first axis. B is perpendicular to q because 2×(−1) + 1×2 = 0.',elements:vectorPicture(1)},
    {title:'Rank by angle',description:'Divide each dot product by both vector lengths. A scores 6/√40 ≈ 0.949, C scores 2/√5 ≈ 0.894, and B scores 0. For this metric, A ranks first even though C is shorter.',elements:vectorPicture(2)},
  ]},
  {id:'retrieval-hnsw-layers',title:'Search a graph from sparse to detailed',summary:'Follow one search through a tiny two-layer proximity graph.',note:'Schematic graph with invented nodes and edges; screen positions illustrate proximity but do not define a measured index. HNSW uses candidate search within layers, not a guaranteed straight route to the exact nearest neighbor.',steps:[
    {title:'Start in a sparse layer',description:'The upper layer contains only a few stored vectors. Search begins at an entry point, A, and compares its neighbors to query q.',elements:graphPicture(0)},
    {title:'Move toward the query',description:'D is closer to q in this illustrative search. Upper-layer links let the search cross a broad region before examining many individual vectors.',elements:graphPicture(1)},
    {title:'Descend at the same vector',description:'D also exists in the base layer. Descending changes the set of available graph connections; it does not re-encode the document or move its embedding.',elements:graphPicture(2)},
    {title:'Explore local candidates',description:'The base layer includes every stored vector. Searching neighbors near D reaches E here. Exploring more candidates can improve recall, at additional work and latency.',elements:graphPicture(3)},
  ]},
  {id:'speculative-token-tree',title:'A draft is a proposal, not committed text',summary:'Watch a proposed suffix pass through target verification, rejection, and correction.',note:'Illustrative stochastic branch: “sat” is accepted and “near” is rejected. The correction “on” represents one possible residual sample; its full distribution is not specified here. Exact sampling preserves the target distribution, not the same random-seed output.',steps:[
    {title:'Draft three candidates',description:'Starting from the committed prefix “The cat”, a cheaper draft proposes “sat near me” sequentially. These tokens have not yet been committed.',elements:speculativePicture(0)},
    {title:'Verify their prefixes together',description:'The target evaluates candidate positions in one causal forward pass. Each position is checked with the same prefix used by the draft; later candidates still depend on earlier proposals.',elements:speculativePicture(1)},
    {title:'Stop at the first rejection',description:'Assume “sat” passes. For “near”, target probability 0.20 divided by draft probability 0.50 gives acceptance probability 0.40. A uniform draw of 0.70 rejects it, so “me” must also be discarded.',elements:speculativePicture(2)},
    {title:'Correct and commit',description:'At the rejected position, sample from the normalized positive residual of target minus draft probabilities. If that sample is “on”, commit “sat on”, reconcile the KV state, and start the next round from that prefix.',elements:speculativePicture(3)},
  ]},
];
