export const categories = [
  { slug: 'foundations', title: 'The foundations', description: 'From tokens and tensors to learning at scale.', icon: 'layers', color: '#b7d9a0' },
  { slug: 'alignment', title: 'Training & alignment', description: 'Turn a next-token predictor into a useful assistant.', icon: 'tune', color: '#ddb58c' },
  { slug: 'architecture', title: 'Advanced architectures', description: 'Attention, experts, and the memory–compute tradeoff.', icon: 'network', color: '#b6b3ee' },
  { slug: 'inference', title: 'Inference systems', description: 'Make every token fast, efficient, and reliable.', icon: 'cpu', color: '#8acbc6' },
  { slug: 'applications', title: 'RAG & agent systems', description: 'Connect models to knowledge and real-world actions.', icon: 'workflow', color: '#a4c6e9' },
  { slug: 'evaluation', title: 'Evaluation & safety', description: 'Measure what matters. Understand what can fail.', icon: 'shield', color: '#e1a6ae' },
  { slug: 'ai-farm', title: 'The corporate AI farm', description: 'A practical, three-part guide to your own inference stack.', icon: 'server', color: '#c5ef82' },
  { slug: 'system-design', title: 'The system design interview', description: 'Bring the mechanisms together. Defend your decisions.', icon: 'compass', color: '#e5cc87' },
] as const;
export const articleOrder = ['transformer-foundations','optimization-generalization','pretraining-data-scaling','post-training-alignment','parameter-efficient-finetuning','efficient-attention','moe-and-parallelism','quantization-and-compression','serving-kv-cache','inference-scheduling','rag-retrieval','agents-tool-systems','evaluation-observability','llm-security','farm-blueprint','farm-deployment','farm-routing-consistency','interview-design'];
