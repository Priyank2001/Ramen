// What each cached run is: one short description per run tag (the `<algo>` part of the results file name), shown
// under the accuracy tiles and as the tooltip of the run menus. Plain data, no build step: edit and reload.
// A tag that matches no rule shows "no description yet"; add a rule below (exact name, or a regex + function).
// Sources: the VARIANTS / RUNS dicts of the scripts named in `src`, and the docstrings in src/methods/.

const RAMEN = 'CLIP test-time adaptation with a memory. Every test image is written, with its entropy-minimisation ' +
  'gradient and its entropy H, into the queue of its zero-shot class. Each query then takes one SignSGD step on a ' +
  'weighted average of the gradients of its nearest memory entries (top-k from every class queue, weight ' +
  'exp(−H)·exp(−β·d)), predicts, and the weights are reset before the next image.';

// cfg/<dataset>/Ramen.yaml (all: SignSGD, lr 0.01)
const RAMEN_CFG = {
  CIFAR10C: 'capacity 7500 per class queue, top-k 50, β 5', CIFAR100C: 'capacity 750, top-k 5, β 5',
  ImageNetC5K: 'capacity 75, top-k 1, β 0', ImageNetVariants: 'capacity 75, top-k 1, β 0',
  DomainNet: 'capacity 300, top-k 10, β 5', EuroSAT: 'capacity 7500, top-k 50, β 5; prompt "a centered satellite photo of {}."',
};

const MODES = {
  mixed: 'mixed stream: all domains shuffled into one stream',
  sequential: 'sequential stream: domains one after another (each shuffled), memory never reset (continual TTA)',
  single: 'single: each domain is its own stream; model and memory reset between domains',
};

const ORACLE = {  // src/methods/RamenOracle.py
  none: ['Oracle harness with no intervention: must equal Ramen exactly (sanity check).', 'sanity'],
  file_true: ['File every image under its TRUE class, so every queue is 100% pure. The stored gradient is unchanged ' +
    '(it still sharpens the zero-shot prediction).', 'oracle'],
  drop_wrong: ['File as Ramen, but never retrieve an entry whose zero-shot prediction was wrong; each queue returns ' +
    'its nearest correct entries instead.', 'oracle'],
  drop_cw: ['Never retrieve an entry that is wrong AND confident (entropy below the running median at write time).',
    'oracle'],
  random_like_wrong: ['Control for oracle_drop_wrong: hide each entry at random with probability = the running ' +
    'zero-shot error rate, so about as many entries are hidden.', 'control'],
  random_like_cw: ['Control for oracle_drop_cw: hide entries at random at the running confident-wrong rate.', 'control'],
};

const CACHE = {  // src/methods/RamenCache.py
  ramen: 'Ramen\'s own write rule (every image, before retrieval, under its zero-shot label): must equal Ramen.',
  all_after: 'Write every image AFTER predicting, under its zero-shot label: control where only the order changes.',
  agree: 'Write an image only if zero-shot CLIP and Ramen predict the same class (after prediction).',
  adapted_label: 'Write every image after predicting, filed under Ramen\'s adapted prediction instead of zero-shot.',
};

const ANCHOR = {  // src/methods/RamenAnchor.py; m = zero-shot top-1 minus top-2 probability margin
  self: v => `up-weight the query's own gradient: α_ii × (1 + ${v}·m)`,
  gate: v => `gate: if the adapted top class differs from zero-shot and the margin m > ${v}, shrink the step until it agrees`,
  tilt: v => `tilt: weight entries from the query's zero-shot queue by exp(${v}·m)`,
  ce: v => `add ${v}·m × the cross-entropy gradient toward the zero-shot pseudo-label`,
};

const QUALITY_METRIC = { niqe: 'NIQE', brisque: 'BRISQUE', both: 'NIQE + BRISQUE (ranks combined)' };
const QUALITY_REG = { match: 'entry quality close to the query\'s (distortion-consistent)', prior: 'entries that look natural (low score)' };

// [regex, (match groups, run) -> {text, kind, src}]; first match wins
const RULES = [
  [/^NoAdapt$/, () => ({ text: 'Zero-shot CLIP, no adaptation: the baseline every TTA run starts from.', kind: 'baseline' })],
  [/^Ramen$/, () => ({ text: RAMEN, kind: 'method' })],
  [/^(NoAdapt|Ramen)_ens7$/, m => ({ text: `${m[1] === 'Ramen' ? 'Ramen' : 'Zero-shot CLIP'} with the repo's default ` +
    '7-prompt template ensemble ("itap of a {}.", …) instead of the satellite prompt: prompt-sensitivity check.',
    kind: 'ablation', src: 'eurosat_ramen.ipynb' })],
  [/^(.+)_p$/, m => ({ text: `Same method as ${m[1]}, re-run with per-sample probabilities saved (for stacking / ` +
    'prior analyses). Predictions should match the cached run.', kind: 'rerun', src: '2026-10-25_stacking/run_probs.py' })],
  [/^RamenLog$/, () => ({ text: 'Ramen with per-sample logging of the retrieved support set. Predictions identical to Ramen.',
    kind: 'rerun', src: 'src/methods/RamenLog.py' })],
  [/^Ramen_cap(\d+)_k(\d+)(?:_b(\d+))?$/, m => ({ text: `Ramen with other cache settings: capacity ${m[1]}, top-k ${m[2]}` +
    (m[3] != null ? `, β ${m[3]}` : '') + ' (another dataset\'s settings, as a sensitivity check).', kind: 'ablation',
    src: 'eurosat_ramen.ipynb' })],
  [/^Ramen_(cdist|fp16)check$/, m => ({ text: `Reproduction check after refactoring the ${m[1] === 'cdist' ? 'distance (cdist)' :
    'fp16 priority'} code: must match the cached mixed-stream Ramen exactly.`, kind: 'sanity', src: '2026-10-26_sequential/run_sequential.py' })],
  [/^Ramen_fp32$/, () => ({ text: 'Ramen with fp32 cache priorities. The default fp16 priorities overflow to inf after ' +
    '65,520 images and freeze full queues; this is the bug fix.', kind: 'fix', src: '2026-10-26_sequential/run_sequential.py' })],
  [/^Ramen_reset$/, () => ({ text: 'Ramen run on each domain separately, with model and memory reset between domains ' +
    '(fresh memory per domain) vs the never-reset sequential run.', kind: 'ablation', src: '2026-10-26_sequential/run_sequential.py' })],

  // RamenActive: each class queue's contribution weighted by the query's zero-shot prediction
  [/^active_all$/, () => ({ text: 'RamenActive with every class queue weighted equally: must equal Ramen (sanity check; ' +
    'also the logged reference run for cache replay).', kind: 'sanity', src: 'src/methods/RamenActive.py' })],
  [/^active_top(\d+)$/, m => ({ text: `Active set = only the query's top-${m[1]} zero-shot classes' queues (others weight 0).`,
    kind: 'variant', src: 'src/methods/RamenActive.py' })],
  [/^active_prob_T([\d.]+)(_fp32)?$/, m => ({ text: `Each class queue weighted by the query's zero-shot probability ` +
    `softmax(logits / ${m[1]}) instead of equally${m[1] === '3' ? ' ("T3", the frozen best setting)' : ''}` +
    (m[2] ? '; with fp32 cache priorities (fp16 overflow fix).' : '.'), kind: 'variant', src: 'src/methods/RamenActive.py' })],

  // RamenPolicy: which memory entries form the active set
  [/^policy_global_k(\d+)$/, m => ({ text: `Active set = the ${m[1]} nearest entries of the WHOLE memory, whatever queue ` +
    'they sit in (classes with many nearby entries vote more).' + (m[1] === '1' ? ' k=1 is per-sample adaptation.' : ''),
    kind: 'variant', src: 'active_set_policies.py' })],
  [/^policy_class_n(\d+)$/, m => ({ text: `Active set = the ${m[1]} nearest entries of every class queue, each queue ` +
    'counting once.' + (m[1] === '1' ? ' n=1 is Ramen\'s rule (top-k 1).' : ''), kind: 'variant', src: 'active_set_policies.py' })],
  [/^policy_zs_T([\d.]+)$/, m => ({ text: `Ramen's active set with each class queue scaled by the query's zero-shot ` +
    `probability softmax(logits / ${m[1]}) (= active_prob_T${m[1]}).`, kind: 'variant', src: 'active_set_policies.py' })],

  // RamenAnchor: zero-shot anchor options; tags like self10, gate0.5, self10+gate0.5
  [/^anchor_off$/, () => ({ text: 'RamenAnchor with every option off: must equal Ramen (sanity check).', kind: 'sanity',
    src: 'anchor_ramen.ipynb' })],
  [/^((self|gate|tilt|ce)[\d.]+)(\+(self|gate|tilt|ce)[\d.]+)*$/, (m, run, tag) => ({
    text: 'Zero-shot anchor (m = zero-shot margin): ' + tag.split('+').map(p => {
      const [, k, v] = p.match(/^(self|gate|tilt|ce)([\d.]+)$/);
      return ANCHOR[k](v);
    }).join('; ') + '.', kind: 'variant', src: 'src/methods/RamenAnchor.py' })],

  // RamenCache: which images enter the memory, under which label, and when
  [/^cache_(ramen|all_after|agree|adapted_label)$/, m => ({ text: CACHE[m[1]], kind: m[1] === 'ramen' ? 'sanity' :
    m[1] === 'all_after' ? 'control' : 'variant', src: 'src/methods/RamenCache.py' })],

  // RamenFix
  [/^clean_g([\d.]+)$/, m => ({ text: `Contamination-aware weighting: α_ij × a_j^${m[1]}, where a_j is how much of entry ` +
    'j\'s own support (when it was the query) lay in its queue; entries whose neighbourhood disagrees with their ' +
    'pseudo-label count less.', kind: 'variant', src: 'src/methods/RamenFix.py' })],
  [/^fp32prio$/, () => ({ text: 'Ramen with fp32 cache priorities only (fp16 overflow fix, via RamenClean with γ = 0).',
    kind: 'fix', src: 'run_fixes.py' })],
  [/^nsgd$/, () => ({ text: 'Normalised SGD instead of SignSGD (same L2 step length).', kind: 'variant', src: 'src/methods/RamenFix.py' })],

  // RamenQuality / RamenNSS: image-quality regularised active set
  [/^quality_l0$/, () => ({ text: 'Quality-regularised active set with λ = 0: must equal Ramen (sanity check).', kind: 'sanity',
    src: 'quality_active_set.py' })],
  [/^quality_(niqe|brisque|both)_(match|prior)_l([\d.]+)(_probT(\d+))?$/, m => ({ text: `Each queue's entry picked by ` +
    `CLIP distance + ${m[3]} × R, R = ${QUALITY_REG[m[2]]}, measured with ${QUALITY_METRIC[m[1]]}` +
    (m[4] ? `; plus zero-shot-probability class weights (T = ${m[5]}).` : '.'), kind: 'variant', src: 'quality_active_set.py' })],
  [/^nss_(niqe|brisque|both)_(white|z)_l([\d.]+)$/, m => ({ text: `Each queue's entry picked by CLIP distance + ${m[3]} × ` +
    `distance between the images' ${m[1] === 'both' ? 'NIQE+BRISQUE (72-d)' : m[1].toUpperCase() + ' (36-d)'} NSS feature ` +
    `vectors (${m[2] === 'white' ? 'running PCA-whitened' : 'running z-scored'}): match the query's distortion type.`,
    kind: 'variant', src: 'nss_active_set.py' })],

  // RamenOracle: true-label control of cache contamination (upper bounds, not methods)
  [/^oracle_(none|file_true|drop_wrong|drop_cw|random_like_wrong|random_like_cw)$/, m => ({ text: ORACLE[m[1]][0] +
    ' Uses TRUE labels: an upper bound, not a method.', kind: ORACLE[m[1]][1], src: '2026-10-23_cache_contamination/run_oracles.py' })],
];

const KINDS = {  // badge text
  baseline: 'baseline', method: 'method', variant: 'variant', ablation: 'ablation', fix: 'bug fix',
  oracle: 'oracle (true labels)', control: 'random control', sanity: 'sanity check (= Ramen)', rerun: 're-run / logging',
};

// algo tag + manifest run -> { text, kind, src, setting }
function describeRun(algo, run) {
  let d = null;
  for (const [re, f] of RULES) {
    const m = algo.match(re);
    if (m) { d = f(m, run, algo); break; }
  }
  d = d || { text: `No description yet for "${algo}": add a rule in confusion_explorer/methods.js.`, kind: null };
  const model = run.model.replace('clip_vitbase', 'CLIP ViT-B/');
  d.setting = [MODES[run.mode] || run.mode, model, `batch ${run.bs}`].join(' · ');
  d.ramen = !/^NoAdapt/.test(algo);  // every other run is Ramen or a modification of it
  if (d.ramen && RAMEN_CFG[run.dataset]) d.setting += ` · base Ramen cfg: ${RAMEN_CFG[run.dataset]}`;
  return d;
}
