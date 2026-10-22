"""
Variable step size for Ramen's adaptation step: how much accuracy is there in choosing the step size?

What Ramen does now
-------------------
For each test sample i, Ramen aggregates its neighbours' cached gradients into one gradient g_i, then takes one
SignSGD step with lr = 0.01:

    w_i = w_0 - 0.01 * sign(g_i)          (w_0 = CLIP's LayerNorm weights, reset after every batch)

So every adapted number moves by exactly 0.01, however strong or weak the evidence is.

What this experiment does
-------------------------
It keeps Ramen's direction, sign(g_i), and only changes how far each sample moves:

    w_i(step) = w_0 - step * sign(g_i)    for every step in STEP_SIZES

step = 0 is zero-shot CLIP, and step = 0.01 is Ramen.

Why one pass over the stream is enough
--------------------------------------
Ramen's memory only holds zero-shot features and gradients at w_0, filed under the zero-shot prediction. The step size
never changes what goes into the memory, so the memory evolves identically for every step size. Each batch therefore
updates the memory and retrieves once, exactly as Ramen does, and is then predicted once per step size. The results
are exact, not an approximation. Cost: one extra forward pass per step size per batch.

Questions it answers
--------------------
1. Is lr = 0.01 the best single step size?                    -> accuracy per step size
2. Does the best step differ by corruption?                   -> best step per corruption (uses labels, diagnostic)
3. Would a per-sample step size help, if we could choose it?  -> oracle: each sample uses its best step in the grid
   If the oracle is barely above the "zero-shot or Ramen" ceiling, choosing the step per sample adds nothing beyond
   choosing whether to adapt at all.

Usage (from new_experiments/, RAMEN env)
-----------------------------------------
    CUDA_VISIBLE_DEVICES=1 python step_size_sweep.py CIFAR10C                     # full stream, seed 0
    CUDA_VISIBLE_DEVICES=1 python step_size_sweep.py CIFAR10C --n_per_domain 50   # quick smoke test
    python step_size_sweep.py CIFAR10C --analyse_only                             # re-print tables from a saved run

In a notebook:
    from step_size_sweep import run_sweep, load_sweep, report
    df, probs = run_sweep('CIFAR10C')     # or: df, probs = load_sweep('CIFAR10C')
    report(df)

Outputs (in results/):
    <run>_StepSweep.pkl        one row per sample, in stream order, with one column pred_<step> per step size
    <run>_StepSweep_probs.npy  softmax for every sample and step: (num_samples, num_steps, num_classes), float16.
                               About 18 MB on CIFAR-10-C, 180 MB on CIFAR-100-C, 2.5 GB on DomainNet
                               (use --no_probs to skip it).

Things to modify
----------------
- STEP_SIZES: the grid.
- RamenStepSweep.apply_step: how the weights move. For example, scale the step per parameter, or use a per-sample
  step chosen from some signal.
- report(): which tables are printed.
"""
import argparse
import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from ramen_analysis import _TaggedDomain, make_experiment  # also puts ../src on the import path
from methods.Ramen import Ramen  # noqa: E402
from methods.losses import softmax_entropy  # noqa: E402
from models.ModelForBySampleTTA import BySampleBatchNorm, BySampleLayerNorm  # noqa: E402

STEP_SIZES = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)
RAMEN_STEP = 0.01      # the lr in cfg/*/Ramen.yaml
RUN_NAME = 'StepSweep'


# ---------------------------------------------------------------------------- the method
class RamenStepSweep(Ramen):
    """Ramen that predicts each batch once per step size, instead of once at lr."""

    def __init__(self, model, datasets, args, step_sizes=STEP_SIZES):
        super().__init__(model, datasets, args)
        self.step_sizes = list(step_sizes)

    @torch.no_grad()
    def apply_step(self, step, B):
        """
        Set the first B samples' adapted weights to w_0 - step * sign(g_i).
        g_i must already be in the parameters' .grad (Ramen puts it there with set_by_sample_grad).
        This is exactly what Ramen's SignSGD optimizer does with lr = step.
        """
        self.model.reset_parameters()  # back to w_0 for every sample
        for m in self.model.model.modules():
            if isinstance(m, (BySampleLayerNorm, BySampleBatchNorm)):
                for p in (m.weight_by_sample, m.bias_by_sample):
                    p[:B].add_(p.grad[:B].sign(), alpha=-step)

    def forward(self, x):
        """Returns {step: logits}. Steps 1-3 are copied from methods/Ramen.py (Ramen.forward); keep them in sync."""
        B = x.shape[0]

        # 1. Zero-shot features, prediction, and each sample's own entropy gradient at w_0.
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        init_preds = logits.argmax(-1)
        loss = self.loss_fn(logits)
        loss.backward()
        grads = self.model.get_by_sample_grad()

        with torch.no_grad():
            # 2. Write the batch into the class-split memory, under its zero-shot prediction.
            #    (fp16 priorities, as in the released code: the overflow bug is kept so step 0.01 == Ramen.)
            priorities = torch.arange(self.counter, self.counter + B, device=self.device, dtype=self.dtype)
            entropies = softmax_entropy(logits, reduction='none')
            self.counter = self.counter + B
            for b in range(B):
                c = init_preds[b]
                self.cache[c].add(keys=feats[b].unsqueeze(0), values=grads[b].unsqueeze(0),
                                  entropies=entropies[b].unsqueeze(0), priorities=priorities[b].unsqueeze(0))

            # 3. Retrieve the top-k neighbours from every class queue and aggregate their gradients.
            retrieved_grads_sum = torch.zeros_like(grads)
            num_class_sum = 0
            for c in range(self.num_classes):
                if self.cache[c].size == 0:
                    continue
                retrieved, _, ent, dist = self.cache[c].query(feats, topk=self.cfg['topk'])
                weights = torch.exp(-ent) * torch.exp(-self.beta * dist)
                retrieved_grads_sum += (retrieved * weights.unsqueeze(2)).sum(dim=1)
                num_class_sum += 1
            self.model.set_by_sample_grad(retrieved_grads_sum / num_class_sum)

            # 4. NEW: one prediction per step size, all from the same aggregated gradient.
            out = {}
            for step in self.step_sizes:
                if step == 0:
                    out[step] = logits.detach()  # no move: zero-shot logits
                    continue
                self.apply_step(step, B)
                out[step] = self.model(x)
            self.model.reset_parameters()

        self.model.optimizer.zero_grad()
        return out


# ---------------------------------------------------------------------------- running
def _paths(exp):
    pkl = exp._cache_path(RUN_NAME)
    return pkl, pkl[:-len('.pkl')] + '_probs.npy'


def run_sweep(dataset, seed=0, n_per_domain=None, step_sizes=STEP_SIZES, save_probs=True):
    """Run the stream once (mixed mode, same sample order as the cached Ramen/NoAdapt runs) and save the results."""
    exp = make_experiment(dataset, seed=seed, n_per_domain=n_per_domain).load()

    # Same order of operations as RamenExperiment.run, so the stream matches the cached runs.
    exp._seed()
    args = SimpleNamespace(config=exp._load_cfg('Ramen'), device=exp.device, max_batch_size=exp.batch_size,
                           use_tbn=False)
    method = RamenStepSweep(deepcopy(exp.clip), exp.datasets, args, step_sizes)
    stream = ConcatDataset([_TaggedDomain(ds, d) for d, ds in enumerate(exp.datasets)])
    loader = DataLoader(stream, batch_size=exp.batch_size, shuffle=True, num_workers=exp.num_workers,
                        pin_memory=True, generator=torch.Generator().manual_seed(exp.seed))

    records, all_probs = [], []
    for batch, (image, label, domain_idx, sample_idx) in enumerate(tqdm(loader, desc=RUN_NAME)):
        out = method(image.to(exp.device))
        probs = torch.stack([out[s].float().softmax(1) for s in step_sizes], dim=1).cpu()  # (B, steps, classes)
        row = {'batch': batch, 'domain_idx': domain_idx.numpy(), 'sample_idx': sample_idx.numpy(),
               'label': label.numpy()}
        for j, s in enumerate(step_sizes):
            row[f'pred_{s:g}'] = probs[:, j].argmax(1).numpy()
        records.append(pd.DataFrame(row))
        if save_probs:
            all_probs.append(probs.numpy().astype(np.float16))

    df = pd.concat(records, ignore_index=True)
    df['position'] = np.arange(len(df))
    df['domain'] = pd.Categorical.from_codes(df['domain_idx'], exp.environments)
    probs = np.concatenate(all_probs) if save_probs else None

    pkl, probs_path = _paths(exp)
    df.to_pickle(pkl)
    if save_probs:
        np.save(probs_path, probs)
    print(f'saved {pkl}')
    check_against_cache(exp, df)
    return df, probs


def load_sweep(dataset, seed=0, n_per_domain=None):
    """Load a saved sweep without re-running (doesn't load the model or data)."""
    exp = make_experiment(dataset, seed=seed, n_per_domain=n_per_domain)
    pkl, probs_path = _paths(exp)
    df = pd.read_pickle(pkl)
    probs = np.load(probs_path, mmap_mode='r') if os.path.exists(probs_path) else None
    return df, probs


def check_against_cache(exp, df):
    """Sanity check: step 0.01 must reproduce the cached Ramen run, step 0 the cached NoAdapt run."""
    keys = ['domain_idx', 'sample_idx']
    for algo, col in (('Ramen', f'pred_{RAMEN_STEP:g}'), ('NoAdapt', 'pred_0')):
        path = exp._cache_path(algo)
        if col not in df or not os.path.exists(path):
            print(f'check: no cached {algo} run to compare {col} with')
            continue
        cached = pd.read_pickle(path)[keys + ['pred']]
        m = df[keys + [col]].merge(cached, on=keys)
        print(f'check: {col} agrees with cached {algo} on {(m[col] == m.pred).mean():.4%} of {len(m)} samples')


# ---------------------------------------------------------------------------- analysis
def steps_in(df):
    return [float(c[len('pred_'):]) for c in df.columns if c.startswith('pred_')]


def domain_avg(correct, df):
    """Average over corruptions/domains of the per-domain accuracy, in % (the paper's metric)."""
    return correct.groupby(df['domain_idx']).mean().mean() * 100


def report(df, noise_floor=None):
    steps = steps_in(df)
    ok = {s: df[f'pred_{s:g}'] == df['label'] for s in steps}  # correct or not, per step
    ramen = domain_avg(ok[RAMEN_STEP], df)

    # 1. One global step size
    print('\n1. One step size for every sample (domain-average accuracy, %)')
    rows = []
    for s in steps:
        r = {'step': s, 'accuracy': domain_avg(ok[s], df), 'vs Ramen': domain_avg(ok[s], df) - ramen}
        if 0.0 in ok:
            r['fixed % (vs zero-shot)'] = (~ok[0.0] & ok[s]).mean() * 100
            r['broken % (vs zero-shot)'] = (ok[0.0] & ~ok[s]).mean() * 100
        rows.append(r)
    print(pd.DataFrame(rows).set_index('step').round(2).to_string())
    if noise_floor:
        print(f'   (a gain counts only above the noise floor, +{noise_floor} pts)')

    # 2. Best step per corruption/domain (diagnostic: uses labels)
    print('\n2. Best single step per domain (uses labels; diagnostic only)')
    per_dom = pd.DataFrame({s: ok[s].groupby(df['domain'], observed=True).mean() * 100 for s in steps})
    t = pd.DataFrame({'best step': per_dom.idxmax(axis=1), 'acc at best': per_dom.max(axis=1),
                      f'acc at {RAMEN_STEP:g}': per_dom[RAMEN_STEP]})
    t['gain'] = t['acc at best'] - t[f'acc at {RAMEN_STEP:g}']
    print(t.round({'acc at best': 2, f'acc at {RAMEN_STEP:g}': 2, 'gain': 2}).to_string())  # keep steps unrounded
    print(f'   average if each domain used its own best step: {per_dom.max(axis=1).mean():.2f} (Ramen {ramen:.2f})')

    # 3. Oracle per-sample step (uses labels): an upper bound for any per-sample step-size rule
    print('\n3. Oracle: each sample uses its best step from the grid (uses labels; an upper bound)')
    def any_of(ss):  # right with at least one of these steps
        return pd.concat([ok[s] for s in ss], axis=1).any(axis=1) if ss else pd.Series(False, index=df.index)

    rows = {f'Ramen (step {RAMEN_STEP:g})': ramen}
    if 0.0 in ok:
        rows['zero-shot or Ramen, per sample (steps 0, 0.01)'] = domain_avg(any_of([0.0, RAMEN_STEP]), df)
    rows['any step > 0, per sample'] = domain_avg(any_of([s for s in steps if s > 0]), df)
    rows['any step in the grid, per sample'] = domain_avg(any_of(steps), df)
    print(pd.Series(rows, name='accuracy').round(2).to_string())

    # 4. Where Ramen is wrong but another step is right: would smaller or larger steps have helped?
    wrong = ~ok[RAMEN_STEP]
    smaller = any_of([s for s in steps if s < RAMEN_STEP])
    larger = any_of([s for s in steps if s > RAMEN_STEP])
    print(f'\n4. Of the samples Ramen gets wrong ({wrong.mean() * 100:.1f}% of all), the share that a different step '
          f'gets right:')
    print(f'   only a smaller step: {(wrong & smaller & ~larger).sum() / wrong.sum() * 100:.1f}%   '
          f'only a larger step: {(wrong & larger & ~smaller).sum() / wrong.sum() * 100:.1f}%   '
          f'both: {(wrong & smaller & larger).sum() / wrong.sum() * 100:.1f}%   '
          f'neither: {(wrong & ~smaller & ~larger).sum() / wrong.sum() * 100:.1f}%')


NOISE_FLOORS = {'CIFAR10C': 0.31, 'CIFAR100C': 0.21}  # seeds 0-4, see findings_and_directions.md

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('dataset', help='CIFAR10C | CIFAR100C | ImageNetC5K | DomainNet')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_per_domain', type=int, default=None, help='random subset per domain (smoke tests)')
    parser.add_argument('--steps', type=float, nargs='+', default=list(STEP_SIZES))
    parser.add_argument('--no_probs', action='store_true', help="don't save the full softmax")
    parser.add_argument('--analyse_only', action='store_true', help='load a saved run instead of running')
    a = parser.parse_args()

    if a.analyse_only:
        df, _ = load_sweep(a.dataset, a.seed, a.n_per_domain)
    else:
        df, _ = run_sweep(a.dataset, a.seed, a.n_per_domain, a.steps, save_probs=not a.no_probs)
    report(df, noise_floor=NOISE_FLOORS.get(a.dataset) if a.n_per_domain is None else None)
