"""
Build the data files for the confusion-matrix explorer (index.html) from the cached per-sample results.

Every `results/<dataset>_<model>_<mode>_n<n>_bs<bs>_s<seed>_<algo>.pkl` (a DataFrame with label / pred / domain)
becomes one sparse table of (domain, true class, predicted class) counts in data/runs/<run>.js, plus one
data/manifest.js listing datasets, class names, domains and runs. The page loads them with <script> tags, so it
works from file:// as well as from a local server.

    ~/miniconda3/envs/RAMEN/bin/python new_experiments/confusion_explorer/build_data.py
    cd new_experiments/confusion_explorer && python -m http.server 8765   # then open localhost:8765

Re-running only rebuilds runs whose .pkl is newer than its data file. data/ is a symlink into results/ (on the
hard disk), like results/ itself.
"""
import argparse
import ast
import base64
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'new_experiments'))
sys.path.insert(0, ROOT)

RUN_RE = re.compile(r'^(?P<dataset>[A-Za-z0-9]+)_(?P<model>clip_[a-z0-9]+)_(?P<mode>mixed|single|sequential)'
                    r'_n(?P<n>[a-z0-9]+)_bs(?P<bs>\d+)_s(?P<seed>\d+)_(?P<algo>.+)\.pkl$')
NOT_RUNS = ('_stats.pkl',)


# ------------------------------------------------------------------ class names (without loading any images)
def _literal_classes(path):
    """`self.classes = [...]` from a dataset module (CIFAR10C / CIFAR100C)."""
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Assign) and any(getattr(t, 'attr', None) == 'classes' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'no class list in {path}')


def _imagenet_classes(data_root):
    from src.datasets.corruption.utils import read_classnames
    names = read_classnames(os.path.join(data_root, 'corruption', 'ImageNet-C', 'classnames.txt'))
    return [names[w] for w in sorted(names)]  # ImageFolder order = sorted wnids


def _folder_classes(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def class_names(dataset, data_root):
    corruption = os.path.join(ROOT, 'src', 'datasets', 'corruption')
    if dataset in ('CIFAR10C', 'CIFAR100C'):
        return _literal_classes(os.path.join(corruption, f'{dataset}.py'))
    if dataset in ('ImageNetC5K', 'ImageNetVariants'):
        return _imagenet_classes(data_root)
    if dataset == 'DomainNet':
        return _folder_classes(os.path.join(data_root, 'domainbed', 'domain_net', 'sketch'))
    if dataset == 'EuroSAT':
        from src.datasets.eurosat import EUROSAT_CLASSNAMES
        return [EUROSAT_CLASSNAMES[c] for c in _folder_classes(os.path.join(data_root, 'EuroSAT', '2750'))]
    raise KeyError(dataset)


def superclasses(dataset):
    """{class name: group} where ramen_analysis defines one (CIFAR-10/100, EuroSAT)."""
    try:
        from ramen_analysis import EXPERIMENTS
    except Exception as e:  # torch / CLIP imports can fail on a machine without the env; groups are optional
        print(f'  (no superclasses: {e})')
        return None
    exp = EXPERIMENTS.get(dataset)
    return dict(exp.superclasses) if exp is not None and exp.superclasses else None


# ------------------------------------------------------------------ one run -> sparse counts
def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode('ascii')


def encode_run(df, n_domains):
    """Per domain: keys = true << 16 | pred (uint32), counts (uint16, or uint32 when a count exceeds 65535)."""
    counts = df.groupby(['domain_idx', 'label', 'pred'], observed=True).size()
    out = []
    for d in range(n_domains):
        if d not in counts.index.get_level_values(0):
            out.append({'k': '', 'c': '', 'w': 2})
            continue
        sub = counts.xs(d, level=0)
        keys = (sub.index.get_level_values(0).to_numpy(np.uint32) << 16) | sub.index.get_level_values(1).to_numpy(np.uint32)
        c = sub.to_numpy()
        w = 2 if c.max() < 65536 else 4
        out.append({'k': _b64(keys.astype('<u4')), 'c': _b64(c.astype('<u2' if w == 2 else '<u4')), 'w': w})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=os.path.join(ROOT, 'new_experiments', 'results'))
    ap.add_argument('--data_root', default='~/data')
    ap.add_argument('--store', default=None,
                    help='where the data files live (default: <results>/confusion_explorer); data/ links to it')
    ap.add_argument('--force', action='store_true', help='rebuild every run')
    args = ap.parse_args()
    data_root = os.path.expanduser(args.data_root)
    store = args.store or os.path.join(args.results, 'confusion_explorer')
    os.makedirs(os.path.join(store, 'runs'), exist_ok=True)
    link = os.path.join(HERE, 'data')
    if not os.path.lexists(link):
        os.symlink(os.path.relpath(store, HERE), link)

    files = sorted(f for f in os.listdir(args.results) if RUN_RE.match(f) and not f.endswith(NOT_RUNS))
    manifest_path = os.path.join(store, 'manifest.json')  # previous manifest: reuse metadata of unchanged runs
    old = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {'runs': {}}
    datasets, runs = {}, {}
    for i, f in enumerate(files):
        m = RUN_RE.match(f).groupdict()
        run_id = f[:-len('.pkl')]
        src, dst = os.path.join(args.results, f), os.path.join(store, 'runs', run_id + '.js')
        fresh = os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src) and run_id in old['runs']
        if fresh and not args.force:
            runs[run_id] = old['runs'][run_id]
            ds_meta = old['datasets'].get(m['dataset'])
            if ds_meta and m['dataset'] not in datasets:
                datasets[m['dataset']] = ds_meta
            continue
        try:
            df = pd.read_pickle(src)
        except Exception as e:
            print(f'skip {f}: {e}')
            continue
        if not isinstance(df, pd.DataFrame) or not {'label', 'pred', 'domain_idx', 'domain'} <= set(df.columns):
            print(f'skip {f}: not a per-sample result table')
            continue
        domains = list(df['domain'].cat.categories) if hasattr(df['domain'], 'cat') else \
            [d for _, d in sorted(df.groupby('domain_idx')['domain'].first().items())]
        first = df.groupby('domain_idx')['domain'].first()
        assert all(domains[d] == name for d, name in first.items()), f'{f}: domain_idx does not match domain'
        ds = m['dataset']
        if ds not in datasets:
            names = class_names(ds, data_root)
            datasets[ds] = {'classes': names, 'domains': domains, 'superclasses': superclasses(ds)}
            print(f'{ds}: {len(names)} classes, domains {domains}')
        n_classes = len(datasets[ds]['classes'])
        assert df['label'].max() < n_classes and df['pred'].max() < n_classes, f'{f}: label out of range'
        assert domains == datasets[ds]['domains'], f'{f}: domains {domains} != {datasets[ds]["domains"]}'

        payload = {'id': run_id, 'domains': encode_run(df, len(domains))}
        with open(dst, 'w') as fh:
            fh.write(f'CMX.add({json.dumps(run_id)},{json.dumps(payload, separators=(",", ":"))});\n')
        per_domain = df.groupby('domain_idx')['correct'].mean().reindex(range(len(domains)))
        runs[run_id] = {'dataset': ds, 'model': m['model'], 'mode': m['mode'], 'n': m['n'], 'bs': int(m['bs']),
                        'seed': int(m['seed']), 'algo': m['algo'], 'samples': int(len(df)),
                        'acc': round(float(df['correct'].mean()) * 100, 3),
                        'acc_domains': [None if np.isnan(a) else round(float(a) * 100, 3) for a in per_domain]}
        print(f'[{i + 1}/{len(files)}] {run_id}: {runs[run_id]["acc"]:.2f}%')

    manifest = {'datasets': datasets, 'runs': runs}
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh)
    with open(os.path.join(store, 'manifest.js'), 'w') as fh:
        fh.write('CMX.manifest = ' + json.dumps(manifest, separators=(',', ':')) + ';\n')
    stale = {f[:-3] for f in os.listdir(os.path.join(store, 'runs'))} - set(runs)
    for run_id in stale:  # the .pkl was deleted
        os.remove(os.path.join(store, 'runs', run_id + '.js'))
    print(f'{len(runs)} runs over {len(datasets)} datasets -> {store}')


if __name__ == '__main__':
    main()
