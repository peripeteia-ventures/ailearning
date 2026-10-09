export const categories = [
  { slug: 'foundations', title: 'How LLMs work', description: 'From raw text to the next token, one mechanism at a time.', icon: 'layers' },
  { slug: 'training', title: 'Training', description: 'How a model learns, what it learns from, and how the run fits on a cluster.', icon: 'trending' },
  { slug: 'alignment', title: 'Post-training', description: 'Turn a next-token predictor into a useful assistant.', icon: 'tune' },
  { slug: 'inference', title: 'Inference & serving', description: 'What actually happens when a model answers, and how to serve many users at once.', icon: 'cpu' },
  { slug: 'architecture', title: 'Efficiency levers', description: 'Spend less memory, bandwidth, and compute per token.', icon: 'zap' },
  { slug: 'applications', title: 'RAG & agent systems', description: 'Connect models to knowledge and real-world actions.', icon: 'workflow' },
  { slug: 'evaluation', title: 'Evaluation & safety', description: 'Measure what matters. Understand what can fail.', icon: 'shield' },
  { slug: 'ai-farm', title: 'The corporate AI farm', description: 'A practical, three-part guide to your own inference stack.', icon: 'server' },
  { slug: 'system-design', title: 'The system design interview', description: 'Bring the mechanisms together. Defend your decisions.', icon: 'compass' },
] as const;
export const articleOrder = ['tokens-and-embeddings','attention-from-scratch','transformer-foundations','optimization-generalization','pretraining-data-scaling','distributed-training','post-training-alignment','parameter-efficient-finetuning','serving-kv-cache','inference-scheduling','efficient-attention','quantization-and-compression','moe-and-parallelism','rag-retrieval','agents-tool-systems','evaluation-observability','llm-security','farm-blueprint','farm-deployment','farm-routing-consistency','interview-design'];
