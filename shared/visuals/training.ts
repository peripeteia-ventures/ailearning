import type { VisualElement as E, VisualLesson, VisualStep } from './types';

const text = (id: string, x: number, y: number, value: string, tone: E['tone'] = 'muted'): E => ({ id, kind: 'text', x, y, text: value, size: 17, tone });
const matrix = (id: string, x: number, y: number, values: (number | string)[][], label: string, tone: E['tone'] = 'blue', extra: Partial<E> = {}): E => ({ id, kind: 'matrix', x, y, values, label, tone, cellW: 68, cellH: 36, ...extra } as E);
const arrow = (id: string, x1: number, y1: number, x2: number, y2: number, label = '', tone: E['tone'] = 'green'): E => ({ id, kind: 'arrow', x1, y1, x2, y2, label, tone });
const box = (id: string, x: number, y: number, w: number, label: string, detail = '', tone: E['tone'] = 'blue'): E => ({ id, kind: 'box', x, y, w, h: 62, label, detail, tone });
const step = (title: string, description: string, elements: E[]): VisualStep => ({ title, description, elements });
const lesson = (id: string, title: string, summary: string, steps: VisualStep[], note = 'Original synthetic example with tiny dimensions; values are explanatory, not measured training results.'): VisualLesson => ({ id, title, summary, note, steps });

const objective = lesson('training-token-objective', 'Find the probability of the observed token', 'A whole vocabulary distribution produces one target loss.', [
  step('Predict a vocabulary row', 'The context “the cat” produces three logits. The observed next token is “sat”.', [
    box('context', 40, 50, 170, 'the cat', 'context tokens'), arrow('forward', 220, 80, 300, 80),
    matrix('scores', 350, 65, [[2, 1, 0]], 'logits z', 'blue', { columnLabels: ['sat', 'ran', 'slept'] }),
    text('target', 350, 160, 'Observed target: sat', 'green'),
  ]),
  step('Normalize across the vocabulary', 'Softmax turns the logits into positive probabilities whose sum is one.', [
    matrix('scores', 55, 105, [[2, 1, 0]], 'logits z'), arrow('normalize', 275, 125, 385, 125, 'softmax'),
    matrix('probs', 430, 105, [[0.665, 0.245, 0.090]], 'probabilities p', 'blue', { columnLabels: ['sat', 'ran', 'slept'] }),
    text('sum', 430, 220, '0.665 + 0.245 + 0.090 = 1'),
  ]),
  step('Select the observed class', 'The target selects the first probability. The loss is −ln(0.6652), about 0.4076 nats.', [
    matrix('probs', 75, 95, [[0.665, 0.245, 0.090]], 'probabilities p', 'blue', { columnLabels: ['sat', 'ran', 'slept'], highlight: [[0, 0]] }),
    arrow('select', 109, 145, 109, 230, 'sat'), box('loss', 55, 250, 220, '−ln(0.6652)', '0.4076 nats', 'green'),
    text('unused', 345, 120, 'One observed target per position'), text('coupled', 345, 158, 'All logits affect its probability'),
  ]),
]);

const crossEntropy = lesson('training-logit-gradient', 'Turn a target into a logit gradient', 'Subtract the one-hot target, then follow the sign of each derivative.', [
  step('Write p and the target', 'The one-hot row puts a one under the correct token and zero elsewhere.', [
    matrix('p', 220, 75, [[0.665, 0.245, 0.090]], 'probability p', 'blue', { columnLabels: ['sat', 'ran', 'slept'] }),
    matrix('y', 220, 170, [[1, 0, 0]], 'one-hot target', 'green'), text('op', 160, 195, '−'),
  ]),
  step('Subtract target from probability', 'Softmax cross-entropy has logit derivative p − onehot(y). The negative entry belongs to the correct token.', [
    matrix('p', 80, 80, [[0.665, 0.245, 0.090]], 'p'), text('op', 300, 104, '−'),
    matrix('y', 350, 80, [[1, 0, 0]], 'target', 'green'), arrow('compute', 345, 135, 345, 210),
    matrix('g', 245, 245, [[-0.335, 0.245, 0.090]], 'gradient ∂ℓ/∂z', 'purple'),
  ]),
  step('Subtracting a negative raises sat', 'For a direct toy update z ← z − 0.1g, the correct logit rises while the other two fall.', [
    matrix('g', 65, 80, [[-0.335, 0.245, 0.090]], 'gradient'),
    matrix('z', 405, 80, [[2, 1, 0]], 'old logits'), arrow('update', 507, 130, 507, 220, '−0.1g'),
    matrix('new', 405, 260, [[2.0335, 0.9755, '−0.009']], 'updated logits', 'green'),
    text('sign', 55, 220, 'Negative derivative → move up', 'green'), text('zero', 55, 260, 'Gradient entries sum to zero'),
  ]),
], 'Rounded three-class example. Real optimizers update shared model parameters, not independent stored logits.');

const graph = (phase: number): E[] => [
  box('w', 40, 60, 110, phase === 3 ? 'w = 0.1' : 'w = 0', 'parameter'),
  box('z', 220, 60, 110, phase === 3 ? 'z = 0.2' : 'z = 0', 'z = w × 2'),
  box('p', 395, 60, 110, phase === 3 ? 'p = .550' : 'p = 0.5', 'sigmoid(z)'),
  box('loss', 565, 60, 120, phase === 3 ? 'ℓ = .598' : 'ℓ = .693', '−ln(p)', 'green'),
  arrow('f1', 155, 91, 210, 91), arrow('f2', 337, 91, 385, 91), arrow('f3', 510, 91, 555, 91),
  ...(phase >= 1 ? [arrow('b1', 620, 170, 450, 170, '∂ℓ/∂p = −2', 'purple'), arrow('b2', 450, 230, 270, 230, '∂p/∂z = 0.25', 'purple')] : []),
  ...(phase >= 2 ? [arrow('b3', 270, 290, 95, 290, '∂z/∂w = 2', 'purple')] : []),
];
const backprop = lesson('training-chain-rule', 'Send derivatives backward through the graph', 'Each reverse edge multiplies an incoming derivative by a local derivative.', [
  step('Forward: compute the scalar loss', 'With x = 2, w = 0 and target y = 1, compute z, then p, then the loss. Save values needed for the reverse pass.', [...graph(0), text('caption', 55, 225, 'Forward values: 0 → 0.5 → 0.6931 nats')]),
  step('Backward: traverse probability and sigmoid', 'The loss sends −1/p = −2. The sigmoid contributes p(1−p) = 0.25; their product is ∂ℓ/∂z = −0.5.', [...graph(1), text('result', 355, 310, '(−2) × 0.25 = −0.5', 'purple')]),
  step('Backward: reach the weight', 'The multiply operation contributes x = 2. Thus ∂ℓ/∂w = (−0.5) × 2 = −1. No parameter has moved yet.', [...graph(2), text('result', 350, 330, '∂ℓ/∂w = −1', 'purple')]),
  step('Optimizer: move, then recompute', 'SGD applies w ← 0 − 0.1(−1) = 0.1. A fresh forward pass lowers the loss to 0.5981.', [...graph(3).filter(e => !e.id.startsWith('b')), text('result', 55, 220, 'Update: w ← w − η × gradient', 'green'), text('fresh', 55, 265, 'New forward values use the new parameter')]),
]);

const landscape = (good: number, bad: number, iteration: number): E[] => {
  const point = (id: string, w: number, tone: E['tone']): E => ({ id, kind: 'circle', x: 350 + 145 * w, y: 305 - 135 * w * w / 2, r: 9, filled: true, tone });
  return [
    { id: 'curve', kind: 'path', d: 'M 89 86.3 Q 350 523.7 611 86.3', tone: 'muted', width: 3 },
    arrow('axis', 70, 305, 645, 305, '', 'muted'), text('w-axis', 655, 310, 'w'), text('yl', 45, 55, 'Loss L(w) = w² / 2'),
    text('left', 90, 335, '−1.8'), text('zero', 345, 335, '0'), text('right', 592, 335, '+1.8'),
    point('good', good, 'green'), point('bad', bad, 'red'),
    text('good-label', 385, 42, `η = 0.5: w = ${good}`, 'green'), text('bad-label', 385, 70, `η = 2.2: w = ${bad}`, 'red'),
    text('iteration', 70, 365, `Update ${iteration} · each dot lies on the same loss surface`),
  ];
};
const descent = lesson('training-loss-landscape', 'Watch a parameter move on the loss surface', 'A gradient gives a local direction; step length determines whether loss falls.', [
  step('Both runs start on the right slope', 'At w = 1 the gradient is +1. Both updates subtract a positive number, moving left.', landscape(1, 1, 0)),
  step('A small step descends; a large step overshoots', 'The light gray parameter reaches 0.5. The red parameter crosses the minimum to −1.2, a higher point on the bowl.', landscape(0.5, -1.2, 1)),
  step('The slope reverses on the other side', 'At −1.2 the red gradient is negative. Subtracting it now moves right, overshooting again to +1.44.', landscape(0.25, 1.44, 2)),
  step('One run converges; the other escapes', 'Light gray reaches 0.125 near the minimum. Red reaches −1.728; its distance and loss grow despite always subtracting the gradient.', landscape(0.125, -1.728, 3)),
], 'Exact scalar quadratic, no noise or momentum. Horizontal position is the parameter, not training time. Red and light gray overlap initially.');

const adam = lesson('training-adamw-state', 'Separate Adam’s memory from weight decay', 'Track two coordinates through moments, normalization, and direct shrinkage.', [
  step('Read the gradient and old weights', 'Two coordinates start with zero moment history. One gradient is positive; the other is negative.', [matrix('theta', 85, 90, [[2, 1]], 'weights θ'), matrix('g', 360, 90, [[0.5, -2]], 'gradient g', 'purple'), text('params', 80, 245, 'β₁ = .9   β₂ = .999   η = .01   λ = .1')]),
  step('Store first and second moments', 'The first moment keeps gradient signs. The second moment stores squared magnitudes; these are optimizer state, not weights.', [matrix('m', 80, 95, [[0.05, -0.2]], 'm = 0.1g', 'purple'), matrix('v', 390, 95, [[0.00025, 0.004]], 'v = 0.001g²', 'amber'), text('correction', 80, 235, 'At step 1: divide m by .1 and v by .001')]),
  step('Correct bias and normalize each coordinate', 'Bias correction gives m̂ = g and v̂ = g². Dividing by √v̂ yields [1, −1], so the adaptive change is [−.01, +.01].', [matrix('mh', 70, 85, [[0.5, -2]], 'm̂'), matrix('vh', 300, 85, [[0.25, 4]], 'v̂', 'amber'), matrix('delta', 485, 235, [['−.01', '+.01']], 'adaptive change', 'green'), arrow('norm', 395, 145, 530, 205, '−η m̂ / √v̂')]),
  step('Shrink the original weights separately', 'Decay contributes −ηλθ = [−.002, −.001]. Add both changes to get [1.988, 1.009]. Decay never entered m or v.', [matrix('theta', 65, 80, [[2, 1]], 'old θ'), matrix('decay', 300, 80, [['−.002', '−.001']], 'decay change', 'amber'), matrix('delta', 525, 80, [['−.01', '+.01']], 'adaptive change', 'purple'), arrow('sum', 365, 135, 365, 220), matrix('new', 297, 260, [[1.988, 1.009]], 'new θ', 'green')]),
], 'First AdamW step with zero moments; epsilon is neglected because these denominators are nonzero. Decay uses the old weights.');

const accumulation = lesson('training-token-accumulation', 'Accumulate sums, then divide by valid tokens', 'Unequal microbatches must have unequal influence on the final mean.', [
  step('Count the valid targets', 'Microbatch A has one target; B has three. Padding does not count. Hold the same weights fixed for both forward passes.', [matrix('a', 150, 80, [[2, 'pad', 'pad']], 'A: token losses', 'blue'), matrix('b', 150, 190, [[4, 4, 4]], 'B: token losses', 'purple'), text('fixed', 420, 150, 'θ held fixed'), text('counts', 420, 200, 'Mₐ = 1; Mᵦ = 3')]),
  step('Add the loss numerators', 'A contributes 2. B contributes 12. The total is 14 over four targets, so the correct mean is 3.5.', [matrix('a', 80, 90, [[2]], 'A sum'), matrix('b', 275, 90, [[12]], 'B sum', 'purple'), arrow('add', 310, 150, 470, 230, 'add / 4'), box('total', 460, 250, 190, 'L = 14 / 4', '3.5 per token', 'green'), text('wrong', 65, 260, '(2 + 4) / 2 = 3 is wrong', 'red')]),
  step('Use the same weights for gradients', 'Illustrative per-token scalar derivatives are 1 for A and 3 for each B token. Accumulate their sum, then divide once by four.', [matrix('ga', 85, 80, [[1]], 'A derivative'), matrix('gb', 275, 80, [[3, 3, 3]], 'B derivatives', 'purple'), box('gmean', 245, 230, 240, 'g = (1 + 9) / 4', '2.5 → one optimizer step', 'green'), arrow('grad', 385, 135, 385, 218)]),
], 'Tiny variable-length batch. The illustrative derivatives are supplied independently; scalar loss values alone do not determine their derivatives.');

const shifted = lesson('training-shifted-targets', 'Build a next-token training tensor', 'Offset each target by one position, then select the loss-bearing entries.', [
  step('Start with one token stream', 'An explicit end token belongs to the stream. Each model position will predict the token immediately to its right.', [matrix('stream', 130, 110, [['the', 'cat', 'sat', 'EOS']], 'token stream', 'blue', { cellW: 105 }), text('causal', 130, 240, 'Each position sees only its prefix')]),
  step('Shift the targets by one token', 'Inputs omit the last token; targets omit the first. The row alignment now directly expresses next-token prediction.', [matrix('input', 160, 70, [['the', 'cat', 'sat']], 'inputs', 'blue', { cellW: 105 }), matrix('target', 160, 220, [['cat', 'sat', 'EOS']], 'targets', 'green', { cellW: 105 }), ...[0, 1, 2].map(i => arrow(`shift-${i}`, 212 + i * 105, 120, 212 + i * 105, 188, 'predict'))]),
  step('Batch rows without scoring padding', 'A shorter stream fills only one target slot. Its two padding targets are excluded from both the loss sum and target count.', [matrix('target', 165, 80, [['cat', 'sat', 'EOS'], ['EOS', 'pad', 'pad']], 'batched targets', 'blue', { cellW: 105, rowLabels: ['A', 'B'] }), matrix('mask', 165, 235, [[1, 1, 1], [1, 0, 0]], 'loss mask: four valid targets', 'green', { cellW: 105, rowLabels: ['A', 'B'] })]),
], 'Token strings stand in for integer token IDs. The example adds an explicit EOS and uses right padding; packing policies vary.');

const trainingState = lesson('training-state-lifetime', 'Separate persistent training state from activations', 'Weights and moments survive updates; intermediates belong to the current computation.', [
  step('Forward creates temporary activations', 'The persistent parameter tensor produces saved hidden values for this microbatch. Adam’s moment tensors are separate state.', [matrix('w', 60, 80, [[1, 2], [3, 4]], 'weights θ'), matrix('m', 250, 80, [[0, 0], [0, 0]], 'Adam m', 'purple'), matrix('v', 450, 80, [[0, 0], [0, 0]], 'Adam v', 'amber'), matrix('h', 255, 255, [[2, 3]], 'saved activation h', 'green'), arrow('forward', 140, 180, 235, 272)]),
  step('Backward consumes intermediates', 'Saved values help compute parameter gradients. Typical training frees this graph after backward; parameter and optimizer tensors remain.', [matrix('w', 60, 80, [[1, 2], [3, 4]], 'weights θ'), matrix('m', 250, 80, [[0, 0], [0, 0]], 'Adam m', 'purple'), matrix('v', 450, 80, [[0, 0], [0, 0]], 'Adam v', 'amber'), matrix('h', 255, 255, [[2, 3]], 'activation: consumed', 'muted', { opacity: 0.4 }), arrow('backward', 245, 270, 145, 190, 'backward', 'purple')]),
  step('A resumable checkpoint needs more than θ', 'Save weights, optimizer tensors, and execution progress. Ordinary update-boundary checkpoints do not need a completed microbatch’s activation graph.', [matrix('w', 60, 80, [[1, 2], [3, 4]], 'save θ'), matrix('m', 250, 80, [[0, 0], [0, 0]], 'save m', 'purple'), matrix('v', 450, 80, [[0, 0], [0, 0]], 'save v', 'amber'), box('progress', 75, 250, 570, 'Also save execution progress', 'schedule · RNG · sampler · counters · configuration', 'green')]),
], 'Schematic tensor values, no memory sizes implied. Moment values are placeholders; master weights and distributed shards are omitted.');

const sftMask = lesson('training-assistant-mask', 'Keep the prompt as context, mask its targets', 'The loss mask and the attention mask have different jobs.', [
  step('Serialize prompt and demonstration', 'A tiny prompt asks “2 + 2 ?”; the demonstration is “4 EOS”. We show five next-token positions after shifting.', [matrix('input', 130, 95, [[2, '+', 2, '?', 4]], 'input tokens', 'blue', { cellW: 90 }), matrix('target', 130, 225, [['+', 2, '?', 4, 'EOS']], 'next-token targets', 'blue', { cellW: 90 })]),
  step('Score assistant targets only', 'The target 4 is predicted at the question-mark position. EOS is predicted after the ground-truth 4. The prompt targets get zero loss weight.', [matrix('target', 130, 70, [['+', 2, '?', 4, 'EOS']], 'targets', 'blue', { cellW: 90, highlight: [[0, 3], [0, 4]] }), matrix('mask', 130, 175, [[0, 0, 0, 1, 1]], 'assistant loss mask', 'green', { cellW: 90 }), text('context', 130, 295, 'Prompt tokens remain visible as causal context')]),
  step('Reduce only the selected losses', 'If p(4)=0.8 and p(EOS)=0.5, their losses sum to 0.916 nats. The mean divides by two assistant targets, not five positions.', [matrix('loss', 130, 85, [['—', '—', '—', 0.223, 0.693]], 'selected token losses', 'green', { cellW: 90 }), box('mean', 230, 235, 280, '(0.223 + 0.693) / 2', '0.458 nats per target', 'green'), arrow('reduce', 490, 140, 400, 218)]),
], 'Simplified one-token arithmetic symbols with no chat-role delimiters. Real chat templates and response boundaries must be explicit.');

const preference = lesson('training-preference-pair', 'Compare a chosen and rejected completion', 'DPO learns from a relative policy-to-reference margin for one shared prompt.', [
  step('Branch the same prompt into two completions', 'The dataset marks one completion chosen and one rejected. This label supplies a comparison, not an absolute truth score.', [box('prompt', 40, 135, 150, 'One prompt'), box('chosen', 350, 55, 270, 'Chosen completion', 'human preference label', 'green'), box('rejected', 350, 225, 270, 'Rejected completion', 'same prompt', 'red'), arrow('branch1', 195, 162, 335, 88), arrow('branch2', 195, 162, 335, 258)]),
  step('Score both with policy and frozen reference', 'Use summed response log probabilities with matching token boundaries. The trainable policy and frozen reference score the same completions.', [matrix('scores', 245, 105, [[-8, -10], [-12, -11]], 'response log probabilities', 'blue', { cellW: 125, columnLabels: ['policy', 'reference'], rowLabels: ['chosen', 'rejected'] }), text('frozen', 335, 260, 'Reference receives no updates')]),
  step('Subtract reference, then compare completions', 'Chosen gains +2 relative to reference; rejected gains −1. Their difference is Δ = 3. With β = 0.2 the preference logit is 0.6.', [matrix('margin', 185, 100, [[2], [-1]], 'log-policy − log-reference', 'purple', { rowLabels: ['chosen', 'rejected'] }), arrow('difference', 275, 140, 400, 140, 'subtract'), box('delta', 425, 110, 210, 'Δ = 2 − (−1)', 'βΔ = 0.6', 'green'), text('loss', 185, 285, '−ln σ(0.6) = 0.437 nats')]),
  step('Backpropagate the relative preference', 'The loss pushes the chosen response log probability up relative to the rejected one. Shared parameters couple these changes; neither absolute probability is guaranteed to move alone.', [box('chosen', 65, 55, 240, 'Chosen log probability', 'loss derivative < 0', 'green'), box('rejected', 405, 55, 240, 'Rejected log probability', 'loss derivative > 0', 'red'), arrow('grad1', 185, 130, 300, 240, 'backward', 'purple'), arrow('grad2', 525, 130, 410, 240, 'backward', 'purple'), box('policy', 245, 265, 220, 'Shared policy θ', 'one parameter update', 'purple')]),
], 'Synthetic sequence scores reproduce the article calculation. Basic offline DPO; reference is frozen, no separate reward model or rollout loop shown.');

export const trainingVisuals: VisualLesson[] = [objective, crossEntropy, backprop, descent, adam, accumulation, shifted, trainingState, sftMask, preference];


