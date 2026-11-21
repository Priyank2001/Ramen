"""
Correctness checks for the gradient probe (DomainNet smoke subset, seed 0, real stream order; GPU, ~1 min).

  G1  per-sample CE gradient: row b of the batched gradient equals the gradient of image b computed ALONE
      (batch of 1): the by-sample LayerNorm trick must not mix images.
  G2  sign convention: one SignSGD step theta - lr * sign(g_ce) raises p(true class) on most images, so a POSITIVE
      cosine with g_ce means "Ramen's step lowers the true-class CE" (to first order).
  G3  decomposition: summing RamenGradientProbe.queue_part (rest + own entry) over every non-empty queue rebuilds
      Ramen's retrieved update, so part_self / part_true / part_zs / part_other are additive shares of cos(g_hat, g_ce).
  G4  own entry: part_self is non-zero exactly for images whose entry was stored, and is parallel to the image's
      own entropy gradient (cos ~ 1; stored in fp16).

    CUDA_VISIBLE_DEVICES=1 python check_gradients.py DomainNet | tee logs/check_gradients_DomainNet.log
"""
import sys

import torch
from torch.utils.data import ConcatDataset, DataLoader

from alignment import cosine
from ga_configs import PROBE_RUN, RUNS
from ga_experiment import GradAlignmentExperiment


def batches(exp, n):
    loader = DataLoader(ConcatDataset(list(exp.datasets)), batch_size=exp.batch_size, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(exp.seed))
    for i, (x, y) in enumerate(loader):
        if i == n:
            return
        yield x, y


def check_per_sample(method, x, y, n=4):
    """G1"""
    model = method.model
    logits = model.classify(model.featurize(x))
    method.loss_fn(logits).backward(retain_graph=True)
    batched = method.oracle_ce_gradient(logits, y)
    model.optimizer.zero_grad()
    worst = 1.0
    for b in range(n):
        lb = model.classify(model.featurize(x[b:b + 1]))
        method.loss_fn(lb).backward(retain_graph=True)      # .grad must exist before oracle_ce_gradient
        alone = method.oracle_ce_gradient(lb, y[b:b + 1])
        model.optimizer.zero_grad()
        c = cosine(batched[b:b + 1], alone).item()
        rel = ((batched[b] - alone[0]).norm() / alone[0].norm()).item()
        worst = min(worst, c)
        print(f'  G1 image {b}: cos(batched row, alone) = {c:.6f}, relative error {rel:.2e}')
    return worst > 0.999


@torch.no_grad()
def p_true(model, x, y):
    return model(x).float().softmax(1).gather(1, y[:, None]).squeeze(1)


def check_sign_convention(method, x, y):
    """G2"""
    model = method.model
    logits = model.classify(model.featurize(x))
    method.loss_fn(logits).backward(retain_graph=True)
    ce = method.oracle_ce_gradient(logits, y)
    p0 = logits.float().softmax(1).gather(1, y[:, None]).squeeze(1).detach()
    model.set_by_sample_grad(ce)
    model.step_and_zero_grad()
    p1 = p_true(model, x, y)
    model.reset_parameters()
    up = int((p1 > p0).sum())
    print(f'  G2 step along -sign(g_ce): p(true) rises on {up} of {len(y)} images, mean {(p1 - p0).mean().item():+.4f}')
    return up > 0.9 * len(y)


def check_decomposition(method, stream):
    """G3: run the probe for a few batches (filling the memory), then rebuild the next batch's update from parts."""
    model = method.model
    for x, y in stream[:-1]:
        method(x, labels=y)
    x, y = stream[-1]
    feats = model.featurize(x)
    logits = model.classify(feats)
    method.loss_fn(logits).backward()
    own = model.get_by_sample_grad()
    model.optimizer.zero_grad()
    zs_pred = logits.argmax(-1)
    self_slot = method.write(feats, own, logits, zs_pred)
    update = method.retrieve(feats, own).float()
    n_queues = method.stats['nonempty_queues'][-1]
    rebuilt = torch.zeros_like(update)
    for c in range(method.num_classes):
        if method.cache[c].size:
            rest, own_c = method.queue_part(feats, torch.full_like(y, c), zs_pred, self_slot, n_queues)
            rebuilt += rest + own_c
    rel = ((rebuilt - update).norm(dim=1) / update.norm(dim=1)).max().item()
    print(f'  G3 {n_queues} non-empty queues: max relative error |sum of parts - update| / |update| = {rel:.2e}; '
          f'min cos = {cosine(rebuilt, update).min().item():.6f}')

    parts = method.decompose(feats, zs_pred, y, self_slot)
    stored = self_slot >= 0
    nonzero = parts['part_self'].norm(dim=1) > 0
    c_own = cosine(parts['part_self'][stored], own[stored]).min().item() if stored.any() else float('nan')
    print(f'  G4 own entry stored for {int(stored.sum())} of {len(y)}; part_self non-zero for {int(nonzero.sum())}; '
          f'same images: {bool((stored == nonzero).all())}; min cos(part_self, own entropy gradient) = {c_own:.6f}')
    return rel < 1e-2, bool((stored == nonzero).all()) and c_own > 0.999


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'DomainNet'
    e = GradAlignmentExperiment(dataset, seed=0, n_per_domain=200, results_dir='smoke')
    e.exp.load()
    exp = e.exp
    method = exp.build_method(*RUNS[PROBE_RUN])
    dev = method.device
    stream = [(x.to(dev), y.to(dev)) for x, y in batches(exp, 5)]
    x, y = stream[0]
    results = {'G1': check_per_sample(method, x, y), 'G2': check_sign_convention(method, x, y)}
    results['G3'], results['G4'] = check_decomposition(method, stream)
    for k, ok in results.items():
        print(f'{k}: {"PASS" if ok else "FAIL"}')
    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
