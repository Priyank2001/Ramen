"""
Real-data correctness checks for the cache-compression runs (CPU, reads only the cached per-image tables and stats).

    ~/miniconda3/envs/RAMEN/bin/python verify.py | tee logs/verify.log

1. cc_off (fp16 codec through the new cache class) == Ramen on every image of the full stream.
2. Every run saw the same stream (same images in the same order) as Ramen.
3. Memory accounting: the cache bytes logged by each run (sum of allocated tensor sizes) equal
   classes x capacity x bytes_per_entry, computed independently here from first principles.
4. Matched controls really are matched: |cache bytes(cc_cap_c) / cache bytes(cc_c) - 1| < 1 queue slot.
5. Table accuracies equal a plain recount from the per-image predictions.
6. One logged update per batch (zero_frac has ceil(images / 100) entries), so the codec ran on every step.
7. No file under src/ or cfg/ changed after this folder was created (the read-only repo rule).
"""
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cc_experiment import CompressionExperiment  # noqa: E402  (first: puts src/ on the path)
from cc_configs import BLOCK, DATASETS, FEAT_DIM, GRAD_DIM  # noqa: E402

CLASSES = {'ImageNetC5K': 1000, 'DomainNet': 345, 'EuroSAT': 10}
NAMES = ['NoAdapt', 'Ramen', 'cc_off', 'cc_int8', 'cc_cap_int8', 'cc_int4', 'cc_cap_int4', 'cc_sign', 'cc_cap_sign']


def value_bytes(codec, D=GRAD_DIM, block=BLOCK):
    """Bytes of one stored gradient, from the format definition (not from the codec classes)."""
    blocks = D // block
    return {'fp16': 2 * D,
            'int8': D + 4 * blocks,                 # 1 byte per value + fp32 scale per block
            'int4': D // 2 + 4 * blocks,            # 2 values per byte + fp32 scale
            'sign': D // 8 + 4 * blocks}[codec]     # 8 signs per byte + fp32 block mean |g|


def entry_bytes(codec):
    return 2 * FEAT_DIM + 2 + 2 + value_bytes(codec)   # fp16 key, fp16 priority, fp16 entropy, value


failures = []


def check(ok, msg):
    print(('PASS  ' if ok else 'FAIL  ') + msg)
    if not ok:
        failures.append(msg)


for ds in DATASETS:
    print(f'== {ds}')
    e = CompressionExperiment(ds, 0)
    for n in NAMES:
        e.run(n)
    al = e.aligned(NAMES)                       # asserts the same sample_idx / domain_idx order for every run
    N = len(al)
    check(True, f'all {len(NAMES)} runs share one stream of {N:,} images')

    same = int((al['cc_off'] == al['Ramen']).sum())
    check(same == N, f'cc_off == Ramen on {same:,} of {N:,} images')

    for n in NAMES[2:]:
        s = e.stats[n]
        expect = CLASSES[ds] * s['max_capacity'] * entry_bytes(s['codec'])
        check(s['cache_bytes'] == expect and s['bytes_per_entry'] == entry_bytes(s['codec']),
              f'{n}: logged cache {s["cache_bytes"]:,} B == {CLASSES[ds]} x {s["max_capacity"]} x '
              f'{entry_bytes(s["codec"]):,} B')
        n_batches = len(e.stats[n]['zero_frac'])
        check(n_batches == math.ceil(N / 100), f'{n}: {n_batches:,} logged updates == ceil({N:,} / 100)')

    for c in ('int8', 'int4', 'sign'):
        a, b = e.stats[f'cc_{c}']['cache_bytes'], e.stats[f'cc_cap_{c}']['cache_bytes']
        slot = CLASSES[ds] * entry_bytes('fp16')
        check(abs(a - b) <= slot, f'cc_cap_{c} bytes {b:,} within one fp16 slot per class ({slot:,} B) of cc_{c} {a:,}')

    t = e.table(NAMES)
    for n in NAMES:
        ok = (al[n] == al['label']).to_numpy()
        dom = al['domain'].to_numpy()
        acc = 100 * np.mean([ok[dom == d].mean() for d in np.unique(dom)])
        check(abs(acc - t.loc[n, 'acc']) < 1e-9 and ok.sum() == t.loc[n, 'correct'],
              f'{n}: recount {ok.sum():,} correct, {acc:.3f} domain-avg == table')

print('== read-only repo')
root = os.path.dirname(os.path.dirname(HERE))
start = min(os.path.getmtime(p) for p in glob.glob(os.path.join(HERE, '*.py')))
touched = [p for d in ('src', 'cfg') for p in glob.glob(os.path.join(root, d, '**', '*'), recursive=True)
           if os.path.isfile(p) and '__pycache__' not in p and os.path.getmtime(p) >= start]
check(not touched, f'no file under src/ or cfg/ modified since this folder was created ({len(touched)} found)')

print(f'\n{"ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED"}')
sys.exit(1 if failures else 0)
