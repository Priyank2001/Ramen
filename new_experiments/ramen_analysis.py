"""
Ramen evaluation + failure analysis on CIFAR-10-C / CIFAR-100-C / ImageNet-C / DomainNet / EuroSAT.

Usage (from a notebook in new_experiments/):
    from ramen_analysis import CIFAR10CExperiment
    exp = CIFAR10CExperiment(tta_mode='mixed').load()
    exp.evaluate()
    exp.plot_domain_accuracy()

Every run is evaluated against a NoAdapt (zero-shot CLIP) baseline on the exact same sample
stream, so the plots can show per-sample where Ramen helps and where it hurts.
"""
import gc
import os
import sys
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from models.get_pretrained_model import get_pretrained_model  # noqa: E402
from datasets import get_dataset_class  # noqa: E402
from methods import get_method_class  # noqa: E402

# ---------------------------------------------------------------------------- palette
TEXT, TEXT_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, AQUA, RED, NEUTRAL = '#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#f0efec'
COLORS = {'Ramen': BLUE, 'NoAdapt': ORANGE}
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']

SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div_red_blue', [RED, NEUTRAL, BLUE])  # red = Ramen worse

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'xtick.color': TEXT_2, 'ytick.color': TEXT_2, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.frameon': False, 'legend.fontsize': 9, 'text.color': TEXT,
})


NOISE_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise']


def expected_calibration_error(df, n_bins=15):
    """ECE in % from a per-sample results frame (columns conf, correct)."""
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(df.conf, edges) - 1, 0, n_bins - 1)
    g = df.assign(bin=b).groupby('bin').agg(conf=('conf', 'mean'), acc=('correct', 'mean'), n=('conf', 'size'))
    return (g.n * (g.acc - g.conf).abs()).sum() / g.n.sum() * 100


def _top_legend(ax, ncol=2):
    """Legend in a row between the title and the plot, so it never covers data."""
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.3)


class _TaggedDomain(Dataset):
    """Wraps one domain's dataset so each item also carries (domain_idx, sample_idx)."""

    def __init__(self, dataset, domain_idx):
        self.dataset, self.domain_idx = dataset, domain_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return x, y, self.domain_idx, i


class RamenExperiment:
    """Base class: load model + data, run TTA methods, collect per-sample records, plot failures."""

    dataset_name = None
    default_model = 'clip_vitbase32'
    superclasses = None            # {class_name: group}, used for error granularity
    watch_classes = ()             # classes whose over/under-prediction summary_metrics tracks
    heatmap_by_superclass = False  # aggregate the class x corruption heatmap by superclass
    env_label = 'corruption'       # what one environment is called in tables and plot titles

    @property
    def _all_envs(self):
        return f'all {self.env_label}s'

    def __init__(self, model=None, tta_mode='mixed', batch_size=100, n_per_domain=None,
                 algos=('NoAdapt', 'Ramen'), seed=0, num_workers=4, severity=5,
                 data_root='~/data', cfg_dir=os.path.join(ROOT, 'cfg'),
                 results_dir=os.path.join(ROOT, 'new_experiments', 'results'), device=None):
        assert tta_mode in ('mixed', 'single'), tta_mode
        self.model_name = model or self.default_model
        self.tta_mode = tta_mode
        self.batch_size = batch_size
        self.n_per_domain = n_per_domain  # None = full dataset, int = random subset per corruption
        self.algos = tuple(algos)
        self.seed = seed
        self.num_workers = num_workers
        self.severity = severity
        self.data_root = os.path.expanduser(data_root)
        self.cfg_dir = cfg_dir
        self.results_dir = results_dir
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.results = {}   # algo -> per-sample DataFrame
        self.features = {}  # algo -> (num_samples, feat_dim) float16 array, row i <-> results[algo].iloc[i]
        self.probs = {}     # algo -> (num_samples, num_classes) float32 softmax, row i <-> results[algo].iloc[i]
        self.method_stats = {}  # name -> method.stats after the run, for methods that keep them (RamenAnchor)

    # ------------------------------------------------------------------ setup
    def _seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def load(self):
        self._seed()
        self.clip, preprocess = get_pretrained_model(SimpleNamespace(model=self.model_name, device=self.device))
        self.datasets = self._build_datasets(preprocess)
        self.environments = list(self.datasets.environments)
        self.classes = list(self.datasets.classes)
        sizes = {len(ds) for ds in self.datasets.datasets}
        size = f'{sizes.pop()} images' if len(sizes) == 1 else \
            f'{sum(len(ds) for ds in self.datasets.datasets)} images in total'
        print(f'{self.dataset_name}: {len(self.environments)} {self.env_label}s x {size}, '
              f'{len(self.classes)} classes, model={self.model_name}, mode={self.tta_mode}')
        return self

    def _build_datasets(self, preprocess):
        datasets = get_dataset_class(self.dataset_name)(root=self.data_root, severity=self.severity,
                                                        transform=preprocess)
        if self.n_per_domain:
            # Same subset of underlying images for every corruption (labels are shared across corruptions)
            n = len(datasets[0])
            idx = np.sort(np.random.default_rng(self.seed).choice(n, self.n_per_domain, replace=False))
            for ds in datasets.datasets:
                ds.X, ds.Y = ds.X[idx], ds.Y[idx]
        return datasets

    def _image(self, domain_idx, sample_idx):
        """Raw (untransformed) test image, for display."""
        return self.datasets[domain_idx].X[sample_idx]

    def _load_cfg(self, algo):
        for path in [os.path.join(self.cfg_dir, self.dataset_name, f'{algo}.yaml'),
                     os.path.join(self.cfg_dir, 'default', f'{algo}.yaml')]:
            if os.path.exists(path):
                with open(path) as f:
                    return yaml.safe_load(f)
        return {}

    def build_method(self, algo, cfg=None):
        args = SimpleNamespace(config={**self._load_cfg(algo), **(cfg or {})}, device=self.device,
                               max_batch_size=self.batch_size, use_tbn=False)
        # Ramen swaps the backbone's norm layers in place, so every method gets its own copy
        return get_method_class(algo)(deepcopy(self.clip), self.datasets, args)

    def _cache_path(self, algo):
        n = self.n_per_domain or 'all'
        name = f'{self.dataset_name}_{self.model_name}_{self.tta_mode}_n{n}_bs{self.batch_size}_s{self.seed}_{algo}.pkl'
        return os.path.join(self.results_dir, name)

    # ------------------------------------------------------------------ running
    def run(self, algo, use_cache=True, capture_features=False, capture_probs=False, name=None, cfg=None):
        """
        Run one TTA method over the stream; returns one row per test sample.
        name, cfg: run a variant of `algo` with these config overrides, stored and cached under `name`
        (e.g. run('RamenAnchor', name='anchor_self10', cfg={'lambda_self': 10})).
        capture_features: also store the image features the final prediction was made from
        (for Ramen: after its per-sample adaptation; for NoAdapt: zero-shot CLIP features).
        capture_probs: also store the full softmax over classes for every sample.
        """
        name = name or algo
        path = self._cache_path(name)
        feat_path = path[:-len('.pkl')] + '_feats.npy'
        probs_path = path[:-len('.pkl')] + '_probs.npy'
        if (use_cache and os.path.exists(path) and (not capture_features or os.path.exists(feat_path))
                and (not capture_probs or os.path.exists(probs_path))):
            self.results[name] = pd.read_pickle(path)
            if os.path.exists(feat_path):
                self.features[name] = np.load(feat_path)
            if os.path.exists(probs_path):
                self.probs[name] = np.load(probs_path)
            if os.path.exists(path[:-len('.pkl')] + '_stats.pkl'):
                self.method_stats[name] = pd.read_pickle(path[:-len('.pkl')] + '_stats.pkl')
            print(f'{name}: loaded cached results from {path}')
            return self.results[name]

        self._seed()
        method = self.build_method(algo, cfg)

        feats, probs, last = [], [], {}
        if capture_features:
            # Both wrappers compute logits as classify(featurize(x)); the last featurize call in a batch
            # is the one behind the returned logits (Ramen calls it once before and once after adapting).
            wrapped = method.model
            featurize = wrapped.featurize

            def featurize_and_keep(x):
                f = featurize(x)
                last['f'] = f
                return f

            wrapped.featurize = featurize_and_keep
        tagged = [_TaggedDomain(ds, d) for d, ds in enumerate(self.datasets)]
        # mixed: one shuffled stream over all corruptions; single: one stream per corruption, reset between
        streams = [ConcatDataset(tagged)] if self.tta_mode == 'mixed' else tagged

        records, batch_counter = [], 0
        for stream in streams:
            # Seeded generator -> every method sees the identical sample order
            loader = DataLoader(stream, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                pin_memory=True, generator=torch.Generator().manual_seed(self.seed))
            desc = name if self.tta_mode == 'mixed' else f'{name} | {self.environments[stream.domain_idx]}'
            for image, label, domain_idx, sample_idx in tqdm(loader, desc=desc):
                if getattr(method, 'records_labels', False):  # e.g. RamenQuality's memory logs them; never predicts
                    logits = method(image.to(self.device), labels=label)
                else:
                    logits = method(image.to(self.device))
                with torch.no_grad():
                    prob = logits.float().softmax(dim=1).cpu()
                    if capture_features:
                        feats.append(last['f'].detach().float().cpu().numpy().astype(np.float16))
                    if capture_probs:
                        probs.append(prob.numpy())
                conf, pred = prob.max(dim=1)
                records.append(pd.DataFrame({
                    'batch': batch_counter,
                    'domain_idx': domain_idx.numpy(),
                    'sample_idx': sample_idx.numpy(),
                    'label': label.numpy(),
                    'pred': pred.numpy(),
                    'conf': conf.numpy(),
                    'true_prob': prob.gather(1, label[:, None]).squeeze(1).numpy(),
                    'entropy': -(prob * prob.clamp_min(1e-12).log()).sum(dim=1).numpy(),
                }))
                batch_counter += 1
            if self.tta_mode == 'single':
                method.reset()
        method.reset()

        df = pd.concat(records, ignore_index=True)
        df['position'] = np.arange(len(df))  # order in which samples arrived in the stream
        df['correct'] = df['label'] == df['pred']
        df['domain'] = pd.Categorical.from_codes(df['domain_idx'], self.environments)

        os.makedirs(self.results_dir, exist_ok=True)
        df.to_pickle(path)
        self.results[name] = df
        if capture_features:
            self.features[name] = np.concatenate(feats)
            np.save(feat_path, self.features[name])
        if capture_probs:
            self.probs[name] = np.concatenate(probs)
            np.save(probs_path, self.probs[name])
        if hasattr(method, 'stats'):
            self.method_stats[name] = method.stats
            pd.to_pickle(method.stats, path[:-len('.pkl')] + '_stats.pkl')
        if hasattr(method, 'save_log'):
            method.save_log(path[:-len('.pkl')])  # e.g. RamenLog's per-sample support sets

        del method
        gc.collect()  # the method's model sits in a reference cycle; free its cache before the next run
        torch.cuda.empty_cache()
        return df

    def evaluate(self, use_cache=True, capture_features=False, capture_probs=False):
        for algo in self.algos:
            self.run(algo, use_cache=use_cache, capture_features=capture_features, capture_probs=capture_probs)
        return self.accuracy_table()

    # ------------------------------------------------------------------ tables
    def _require_baseline(self):
        missing = {'NoAdapt', 'Ramen'} - set(self.results)
        if missing:
            raise RuntimeError(f'Needs results for {missing}; run exp.evaluate() with both algos.')

    def accuracy_table(self):
        acc = pd.DataFrame({a: df.groupby('domain_idx')['correct'].mean() for a, df in self.results.items()}) * 100
        acc.index = [self.environments[i] for i in acc.index]
        if {'NoAdapt', 'Ramen'} <= set(acc.columns):
            acc['delta'] = acc['Ramen'] - acc['NoAdapt']
        acc.loc['average'] = acc.mean()
        return acc.round(2)

    def paired(self):
        """Ramen rows joined with NoAdapt's prediction for the same sample, plus a flip outcome."""
        self._require_baseline()
        keys = ['domain_idx', 'sample_idx']
        p = self.results['Ramen'].merge(self.results['NoAdapt'][keys + ['pred', 'conf', 'entropy', 'correct']],
                                        on=keys, suffixes=('', '_base'))
        p['outcome'] = np.select(
            [p.correct & ~p.correct_base, ~p.correct & p.correct_base, p.correct & p.correct_base],
            ['fixed', 'broken', 'both right'], default='both wrong')
        return p

    def error_granularity(self):
        """Share of errors that stay within the true class's superclass (fine-grained confusions)."""
        if self.superclasses is None:
            raise RuntimeError(f'{type(self).__name__} defines no superclasses.')
        group = np.array([self.superclasses[c] for c in self.classes])
        out = {}
        for algo, df in self.results.items():
            err = df[~df.correct]
            within = group[err.label.values] == group[err.pred.values]
            out[algo] = pd.Series(within, index=err.index).groupby(err.domain_idx.values).mean() * 100
        table = pd.DataFrame(out)
        table.index = [self.environments[i] for i in table.index]
        table.loc['average'] = table.mean()
        return table.round(1).add_suffix(' % errors within superclass')

    def gate_sweep(self, taus=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
        """
        Post-hoc "keep zero-shot when it is confident" gate: for each tau, use NoAdapt's prediction where its
        confidence >= tau and Ramen's elsewhere. Exact for a gate that only skips a sample's own update,
        because the memory (written before adaptation) is unchanged. No re-run needed.
        """
        p = self.paired()
        return self._gate_table(p, {tau: p.conf_base >= tau for tau in taus})

    def entropy_gate_sweep(self, taus=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
        """
        Post-hoc "don't adapt when zero-shot entropy is already low" gate: for each tau, use NoAdapt's prediction
        where its entropy <= tau * ln(C) and Ramen's elsewhere. ln(C) is the largest possible entropy, so tau is a
        fraction of it and the same taus work for any number of classes (EATA uses 0.4 * ln C as its threshold).
        Exact without a re-run, for the same reason as gate_sweep.
        """
        p = self.paired()
        max_entropy = np.log(len(self.classes))
        table = self._gate_table(p, {tau: p.entropy_base <= tau * max_entropy for tau in taus})
        table.insert(0, 'entropy threshold', (table.index.to_numpy() * max_entropy).round(3))
        return table

    @staticmethod
    def _gate_table(p, gates):
        """One row per threshold: use NoAdapt's prediction where the gate is on (g True), Ramen's elsewhere."""
        rows = []
        for tau, g in gates.items():
            rows.append({
                'tau': tau,
                'gated %': g.mean() * 100,
                'NoAdapt acc in gate': p.correct_base[g].mean() * 100,
                'Ramen acc in gate': p.correct[g].mean() * 100,
                'gated total acc': np.where(g, p.correct_base, p.correct).mean() * 100,
                'broken saved': int((g & p.correct_base & ~p.correct).sum()),
                'fixed lost': int((g & ~p.correct_base & p.correct).sum()),
            })
        table = pd.DataFrame(rows).set_index('tau').round(2)
        table.attrs['Ramen'] = p.correct.mean() * 100
        table.attrs['oracle'] = (p.correct | p.correct_base).mean() * 100  # ceiling for any zs/Ramen selector
        return table

    def summary_metrics(self, algo='Ramen'):
        """
        Scalar metrics a method (or a future fix) is judged on, as one Series per run:
        accuracy (avg / noise / other / per corruption), flips vs NoAdapt, class-distribution
        over-correction, and calibration. Needs NoAdapt results for the flip metrics.
        """
        r = self.results[algo]
        acc = r.groupby('domain_idx')['correct'].mean() * 100
        names = [self.environments[i] for i in acc.index]
        is_noise = np.array([n in NOISE_CORRUPTIONS for n in names])
        m = {'avg': acc.mean(), 'noise': acc[is_noise].mean(), 'other': acc[~is_noise].mean()}

        if algo == 'Ramen' and 'NoAdapt' in self.results:
            outcome = self.paired()['outcome']
            m['fixed %'] = (outcome == 'fixed').mean() * 100
            m['broken %'] = (outcome == 'broken').mean() * 100

        k = len(self.classes)
        true = np.bincount(r.label, minlength=k) / len(r)
        pred = np.bincount(r.pred, minlength=k) / len(r)
        m['TV(pred, true)'] = 0.5 * np.abs(pred - true).sum()
        for c in self.watch_classes:
            i = self.classes.index(c)
            m[f'{c} overshoot %'] = (pred[i] / true[i] - 1) * 100 if true[i] > 0 else np.nan
        m['ECE'] = expected_calibration_error(r)
        m.update({f'acc {n}': a for n, a in zip(names, acc.values)})
        return pd.Series(m)

    def class_shares(self, algo='Ramen'):
        """Predicted share / true share - 1, per class, in % (positive = over-predicted)."""
        r = self.results[algo]
        k = len(self.classes)
        true = np.bincount(r.label, minlength=k).astype(float)
        true[true == 0] = np.nan  # classes absent from a small subset
        return pd.Series((np.bincount(r.pred, minlength=k) / true - 1) * 100, index=self.classes)

    def _select(self, algo, domain):
        df = self.results[algo]
        if domain is None:
            return df
        return df[df.domain_idx == self.environments.index(domain)]

    # ------------------------------------------------------------------ plots
    def plot_domain_accuracy(self):
        """Dumbbell: NoAdapt -> Ramen accuracy per corruption, worst Ramen corruption on top."""
        acc = self.accuracy_table().drop('average').sort_values(self.algos[-1])
        y = np.arange(len(acc))
        fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(acc) + 1.4))

        has_both = {'NoAdapt', 'Ramen'} <= set(acc.columns)
        if has_both:
            ax.hlines(y, acc['NoAdapt'], acc['Ramen'], color=MUTED, lw=2, zorder=1)
        for algo in self.algos:
            ax.scatter(acc[algo], y, s=70, color=COLORS[algo], edgecolor=SURFACE, linewidth=2, zorder=3,
                       label=f'{algo} (avg {acc[algo].mean():.1f}%)')
        if has_both:
            x_text = acc[['NoAdapt', 'Ramen']].values.max() + 2
            for yi, d in zip(y, acc['delta']):
                ax.text(x_text, yi, f'{d:+.1f}', va='center', fontsize=9,
                        color=TEXT if d < 0 else TEXT_2, fontweight='bold' if d < 0 else 'normal')
            ax.set_xlim(right=x_text + 6)

        ax.set_yticks(y, acc.index)
        ax.invert_yaxis()
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title(f'{self.dataset_name}: accuracy per {self.env_label} (right column = Ramen - NoAdapt, pts)', pad=24)
        _top_legend(ax)
        fig.tight_layout()
        return fig

    def plot_flips(self):
        """Per corruption: % samples Ramen fixed (NoAdapt wrong) vs broke (NoAdapt right)."""
        p = self.paired()
        n = p.groupby('domain_idx').size()
        counts = p.groupby('domain_idx')['outcome'].value_counts().unstack(fill_value=0)
        counts = counts.reindex(columns=['fixed', 'broken'], fill_value=0)
        fixed, broken = counts['fixed'] / n * 100, counts['broken'] / n * 100
        net = (fixed - broken).sort_values()
        fixed, broken = fixed[net.index], broken[net.index]
        y = np.arange(len(net))

        fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(net) + 1.4))
        ax.barh(y, fixed, height=0.62, color=BLUE, label='fixed: NoAdapt wrong -> Ramen right')
        ax.barh(y, -broken, height=0.62, color=RED, label='broken: NoAdapt right -> Ramen wrong')
        ax.axvline(0, color=TEXT_2, lw=1)
        x_text = fixed.max() + 0.5
        for yi, v in zip(y, net):
            ax.text(x_text, yi, f'net {v:+.1f}', va='center', fontsize=9,
                    color=TEXT if v < 0 else TEXT_2, fontweight='bold' if v < 0 else 'normal')
        ax.set_xlim(-broken.max() * 1.15, x_text + max(2.5, fixed.max() * 0.35))
        ax.set_yticks(y, [self.environments[i] for i in net.index])
        ax.grid(axis='y', visible=False)
        ax.set_xlabel(f'% of samples in the {self.env_label}')
        ax.set_title(f'{self.dataset_name}: per-sample flips vs NoAdapt (worst net on top)', pad=24)
        _top_legend(ax)
        fig.tight_layout()
        return fig

    def plot_class_heatmap(self, metric='delta', by_superclass=None, max_rows=25):
        """Accuracy per (class or superclass) x corruption. metric: 'delta' | 'Ramen' | 'NoAdapt'."""
        by_superclass = self.heatmap_by_superclass if by_superclass is None else by_superclass
        groups = np.array([self.superclasses[c] for c in self.classes] if by_superclass else self.classes)

        def grouped_acc(algo):
            d = self.results[algo]
            return d.assign(group=groups[d.label.values]).pivot_table(
                index='group', columns='domain_idx', values='correct', aggfunc='mean') * 100

        if metric == 'delta':
            self._require_baseline()
            mat = grouped_acc('Ramen') - grouped_acc('NoAdapt')
            lim = max(np.nanmax(np.abs(mat.values)), 1e-6)
            cmap, norm, cbar_label = DIV, TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), 'Ramen - NoAdapt (pts)'
            dark = lambda v: abs(v) > 0.6 * lim
        else:
            mat = grouped_acc(metric)
            cmap, norm, cbar_label = SEQ, plt.Normalize(0, 100), f'{metric} accuracy (%)'
            dark = lambda v: v > 55

        mat = mat.loc[mat.mean(axis=1).sort_values().index[:max_rows]]  # worst rows on top
        mat.columns = [self.environments[i] for i in mat.columns]

        fig, ax = plt.subplots(figsize=(0.55 * mat.shape[1] + 3.5, 0.36 * mat.shape[0] + 2.2))
        im = ax.imshow(mat.values, cmap=cmap, norm=norm, aspect='auto')
        for (i, j), v in np.ndenumerate(mat.values):
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7, color='white' if dark(v) else TEXT)
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha='right')
        ax.set_yticks(range(mat.shape[0]), mat.index)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)
        level = 'superclass' if by_superclass else 'class'
        shown = f' (worst {max_rows} shown)' if len(np.unique(groups)) > max_rows else ''
        ax.set_title(f'{self.dataset_name}: {cbar_label} by {level} x {self.env_label}{shown}')
        fig.tight_layout()
        return fig

    def plot_confusion(self, algo='Ramen', domain=None):
        """Row-normalised confusion matrix of errors (diagonal shown as text, kept out of the colour scale)."""
        d = self._select(algo, domain)
        k = len(self.classes)
        cm = pd.crosstab(d.label, d.pred).reindex(index=range(k), columns=range(k), fill_value=0).values
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
        off = cm.copy()
        np.fill_diagonal(off, np.nan)
        cmap = SEQ.copy()
        cmap.set_bad(NEUTRAL)
        vmax = np.nanmax(off)

        fig, ax = plt.subplots(figsize=(0.6 * k + 2.5, 0.6 * k + 1.5))
        im = ax.imshow(off, cmap=cmap, vmin=0, vmax=vmax)
        for (i, j), v in np.ndenumerate(cm):
            if i == j:
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8, color=TEXT_2, fontstyle='italic')
            elif v >= 1:
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if v > 0.55 * vmax else TEXT)
        ax.set_xticks(range(k), self.classes, rotation=45, ha='right')
        ax.set_yticks(range(k), self.classes)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='% of true class predicted as column')
        ax.set_title(f'{algo} confusions on {domain or self._all_envs} (diagonal = per-class accuracy %)')
        fig.tight_layout()
        return fig

    def plot_online_accuracy(self, window=None):
        """Rolling accuracy along the test stream (top) and Ramen - NoAdapt gap (bottom)."""
        n_total = len(next(iter(self.results.values())))
        single = self.tta_mode == 'single'
        window = window or (max(20, n_total // len(self.environments) // 10) if single else max(200, n_total // 75))

        def rolling(df, col):
            s = df.sort_values('position')
            v = s[col].astype(float)
            roll = lambda x: x.rolling(window, min_periods=window // 2).mean()
            r = v.groupby(s['domain_idx'].values).transform(roll) if single else roll(v)  # no smoothing across resets
            return s['position'].values, r.values * 100

        has_both = {'NoAdapt', 'Ramen'} <= set(self.results)
        fig, axes = plt.subplots(2 if has_both else 1, 1, sharex=True, squeeze=False,
                                 figsize=(10.5, 6 if has_both else 4), height_ratios=[2, 1] if has_both else [1])
        ax1 = axes[0, 0]
        for algo in self.algos:
            x, r = rolling(self.results[algo], 'correct')
            ax1.plot(x, r, lw=2, color=COLORS[algo], label=algo)
            last = np.flatnonzero(~np.isnan(r))[-1]
            ax1.text(x[last] + n_total * 0.005, r[last], algo, va='center', fontsize=9, color=TEXT_2)
        ax1.set_ylabel(f'accuracy (%), window={window}')
        _top_legend(ax1)
        ax1.set_xlim(0, n_total * 1.06)

        if has_both:
            ax2 = axes[1, 0]
            p = self.paired()
            p['gain'] = p.correct.astype(int) - p.correct_base.astype(int)
            x, g = rolling(p, 'gain')
            ax2.fill_between(x, g, 0, where=g >= 0, color=BLUE, alpha=0.35, lw=0, interpolate=True)
            ax2.fill_between(x, g, 0, where=g < 0, color=RED, alpha=0.35, lw=0, interpolate=True)
            ax2.plot(x, g, lw=1, color=TEXT_2)
            ax2.axhline(0, color=TEXT_2, lw=1)
            ax2.set_ylabel('Ramen - NoAdapt (pts)')

        if single:
            pos = self.results[self.algos[0]].groupby('domain_idx')['position'].agg(['min', 'max'])
            for ax in axes[:, 0]:
                for start in pos['min'].iloc[1:]:
                    ax.axvline(start, color=MUTED, lw=0.8, ls='--')
            axes[-1, 0].set_xticks((pos['min'] + pos['max']) / 2, [self.environments[d] for d in pos.index],
                                   rotation=45, ha='right')
            axes[-1, 0].set_xlabel(f'{self.env_label}, in stream order')
        else:
            axes[-1, 0].set_xlabel('samples seen in the test stream')
        mode = (f'per {self.env_label}, reset at dashed lines' if single
                else f'{self._all_envs} mixed in one stream')
        ax1.set_title(f'{self.dataset_name}: online accuracy ({mode})', pad=24)
        fig.tight_layout()
        return fig

    def plot_calibration(self, n_bins=15):
        """Reliability diagram per method, and Ramen's confidence on correct vs wrong predictions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        edges = np.linspace(0, 1, n_bins + 1)
        ax1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--')
        for algo in self.algos:
            d = self.results[algo]
            b = np.clip(np.digitize(d.conf, edges) - 1, 0, n_bins - 1)
            g = d.assign(bin=b).groupby('bin').agg(conf=('conf', 'mean'), acc=('correct', 'mean'), n=('conf', 'size'))
            ece = (g.n * (g.acc - g.conf).abs()).sum() / g.n.sum() * 100
            ax1.plot(g.conf, g.acc, lw=2, marker='o', ms=7, mec=SURFACE, mew=1.5, color=COLORS[algo],
                     label=f'{algo} (ECE {ece:.1f}%)')
        ax1.set(xlim=(0, 1), ylim=(0, 1), xlabel='confidence', ylabel='accuracy')
        ax1.set_title('Reliability (below diagonal = overconfident)', pad=24)
        _top_legend(ax1, ncol=1 if len(self.algos) > 1 else 2)

        algo = self.algos[-1]
        d = self.results[algo]
        ax2.hist(d.conf[d.correct], bins=edges, density=True, color=MUTED, alpha=0.45, label='correct')
        ax2.hist(d.conf[~d.correct], bins=edges, density=True, histtype='step', lw=2, color=RED, label='wrong')
        ax2.set(xlim=(0, 1), xlabel='confidence', ylabel='density (log)', yscale='log')
        ax2.set_title(f'{algo}: confidence of correct vs wrong predictions', pad=24)
        _top_legend(ax2)
        fig.tight_layout()
        return fig

    def plot_prediction_bias(self, domain=None, top_k=20):
        """Predicted share minus true share per class; large positive bars = predictions collapsing onto a class."""
        k = len(self.classes)
        frames = {a: self._select(a, domain) for a in self.algos}
        any_df = next(iter(frames.values()))
        true_share = np.bincount(any_df.label, minlength=k) / len(any_df) * 100
        bias = pd.DataFrame({a: np.bincount(d.pred, minlength=k) / len(d) * 100 - true_share
                             for a, d in frames.items()}, index=self.classes)
        key = self.algos[-1]
        if k > top_k:
            bias = bias.loc[bias[key].abs().nlargest(top_k).index]
        bias = bias.sort_values(key, ascending=False)

        y = np.arange(len(bias))
        h = 0.8 / len(self.algos)
        fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(bias) * max(1, len(self.algos) / 1.5) + 1.6))
        for i, algo in enumerate(self.algos):
            ax.barh(y + (i - (len(self.algos) - 1) / 2) * h, bias[algo], height=h * 0.92,
                    color=COLORS[algo], label=algo)
        ax.axvline(0, color=TEXT_2, lw=1)
        ax.set_yticks(y, bias.index)
        ax.invert_yaxis()
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('predicted share - true share (pts)   [> 0: over-predicted]')
        shown = f', top {top_k} by |{key} bias|' if k > top_k else ''
        ax.set_title(f'{self.dataset_name}: class prediction bias on {domain or self._all_envs}{shown}', pad=24)
        _top_legend(ax)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ probability distributions
    def _probs_for(self, algo, domain=None):
        """Per-sample rows of one method and the matching (n, C) softmax rows."""
        if algo not in self.probs:
            raise RuntimeError(f'No stored probabilities for {algo}; run exp.evaluate(capture_probs=True).')
        d = self._select(algo, domain)
        return d, self.probs[algo][d.index.values]

    def confused_pairs(self, domain=None, top_k=10):
        """Most frequent (true -> predicted) errors as % of the true class, per method, sorted by their mean."""
        k = len(self.classes)
        rates = {}
        for algo in self.algos:
            d = self._select(algo, domain)
            cm = pd.crosstab(d.label, d.pred, normalize='index').reindex(
                index=range(k), columns=range(k), fill_value=0) * 100
            s = cm.stack()
            rates[algo] = s[s.index.get_level_values(0) != s.index.get_level_values(1)]
        t = pd.DataFrame(rates)
        t['mean'] = t.mean(axis=1)
        t = t.nlargest(top_k, 'mean')
        t.index = pd.MultiIndex.from_tuples([(self.classes[a], self.classes[b]) for a, b in t.index],
                                            names=['true', 'predicted as'])
        return t.round(2)

    def most_confused_classes(self, domain=None, n=4):
        """True classes with the highest error rate, averaged over the methods."""
        err = pd.DataFrame({a: 1 - self._select(a, domain).groupby('label')['correct'].mean() for a in self.algos})
        return [self.classes[i] for i in err.mean(axis=1).nlargest(n).index]

    def plot_class_probs(self, classes=None, domain=None, n=4, show_k=10, cols=2):
        """
        For each true class, the mean softmax over all classes (NoAdapt vs Ramen), averaged over that class's samples.
        Mass that sits on another class is what the model confuses it with. classes=None picks the n most confused.
        show_k: number of classes on the x-axis (the true class plus the most probable others).
        """
        classes = self._class_ids(classes or self.most_confused_classes(domain, n))
        rows = int(np.ceil(len(classes) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.4 * rows), squeeze=False)
        h = 0.8 / len(self.algos)
        for ax, c in zip(axes.flat, classes):
            mean, acc = {}, {}
            for algo in self.algos:
                d, p = self._probs_for(algo, domain)
                mask = (d.label == c).values
                mean[algo], acc[algo] = p[mask].mean(0), d.correct.values[mask].mean() * 100
            others = pd.Series(np.max(list(mean.values()), axis=0)).drop(c).nlargest(show_k - 1).index
            shown = [c] + sorted(others)
            x = np.arange(len(shown))
            for i, algo in enumerate(self.algos):
                ax.bar(x + (i - (len(self.algos) - 1) / 2) * h, mean[algo][shown], width=h * 0.92,
                       color=COLORS[algo], label=algo)
            ax.set_xticks(x, [self.classes[j] for j in shown], rotation=45, ha='right')
            ax.get_xticklabels()[0].set_fontweight('bold')
            ax.set_ylim(0, 1 if len(self.classes) <= 20 else None)  # 100 classes: mass is spread thin, autoscale
            ax.grid(axis='x', visible=False)
            ax.set_ylabel('mean probability')
            accs = '  '.join(f'{a} {acc[a]:.1f}%' for a in self.algos)
            ax.set_title(f'true: {self.classes[c]}   (acc {accs})', fontsize=10)
        for ax in axes.flat[len(classes):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=len(self.algos))
        fig.suptitle(f'{self.dataset_name}: mean predicted distribution for the most confused classes, '
                     f'{domain or self._all_envs} (bold = true class)', x=0.01, ha='left',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    def plot_confusion_margin(self, pairs=None, domain=None, top_k=4, cols=2, bins=40):
        """
        For each confused pair (A -> B): histogram over the samples of true class A of p(A) - p(B).
        Below 0 the confuser wins; near 0 the model is torn between the two; far below 0 it is confidently wrong.
        pairs=None uses the top_k pairs from confused_pairs().
        """
        if pairs is None:
            pairs = list(self.confused_pairs(domain, top_k).index)
        rows = int(np.ceil(len(pairs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.0 * rows), squeeze=False)
        edges = np.linspace(-1, 1, bins + 1)
        for ax, (a, b) in zip(axes.flat, pairs):
            ia, ib = self._class_ids([a, b])
            notes = []
            for algo in self.algos:
                d, p = self._probs_for(algo, domain)
                m = p[(d.label == ia).values]
                margin = m[:, ia] - m[:, ib]
                ax.hist(margin, bins=edges, histtype='step', lw=2, color=COLORS[algo], label=algo)
                notes.append(f'{algo}: {(margin < 0).mean() * 100:.1f}% below 0')
            ax.axvline(0, color=TEXT_2, lw=1)
            ax.set_yscale('log')
            ax.set_xlim(-1, 1)
            ax.set_xlabel(f'p({self.classes[ia]}) - p({self.classes[ib]})')
            ax.set_ylabel('samples (log)')
            ax.set_title(f'true {self.classes[ia]}, confuser {self.classes[ib]}', fontsize=10)
            ax.text(0.02, 0.97, '\n'.join(notes), transform=ax.transAxes, va='top', fontsize=8, color=TEXT_2)
        for ax in axes.flat[len(pairs):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=len(self.algos))
        fig.suptitle(f'{self.dataset_name}: true-class vs confuser probability, {domain or self._all_envs} '
                     f'(< 0 = confuser wins)', x=0.01, ha='left', fontsize=12, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    def pick_failure(self, pair=None, kind='broken', domain=None, rank=0):
        """
        (domain_idx, sample_idx) of one Ramen failure where true -> predicted is `pair` (default: top confused pair).
        kind: 'broken' (NoAdapt was right) or 'wrong' (any Ramen error). rank 0 = the one Ramen is most confident on.
        """
        p = self.paired()
        if domain is not None:
            p = p[p.domain_idx == self.environments.index(domain)]
        ia, ib = self._class_ids(list(pair or self.confused_pairs(domain, top_k=1).index[0]))
        m = p[(p.label == ia) & (p.pred == ib)]
        if kind == 'broken':
            m = m[m.correct_base]
        if len(m) <= rank:
            raise ValueError(f'only {len(m)} {kind} {self.classes[ia]} -> {self.classes[ib]} samples')
        r = m.sort_values('conf', ascending=False).iloc[rank]
        return int(r.domain_idx), int(r.sample_idx)

    def plot_sample_probs(self, domain_idx, sample_idx, show_k=None):
        """
        One test image next to each method's predicted distribution over the classes.
        show_k: number of classes on the x-axis (the true class plus the most probable others);
        None = all classes when there are at most 20, else 15.
        """
        rows = {a: self.results[a].index[(self.results[a].domain_idx == domain_idx)
                                         & (self.results[a].sample_idx == sample_idx)][0] for a in self.algos}
        label = int(self.results[self.algos[0]].label[rows[self.algos[0]]])
        k = len(self.classes)
        show_k = show_k or (k if k <= 20 else 15)
        top = np.max([self.probs[a][rows[a]] for a in self.algos], axis=0)
        shown = list(range(k)) if show_k >= k else [label] + sorted(
            pd.Series(top).drop(label).nlargest(show_k - 1).index)
        fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [1, 3.2]})
        ax_img.imshow(self._image(domain_idx, sample_idx), interpolation='nearest')
        ax_img.axis('off')
        ax_img.set_title(f'{self.environments[domain_idx]} #{sample_idx}\ntrue: {self.classes[label]}',
                         fontsize=10, loc='center')

        x = np.arange(len(shown))
        h = 0.8 / len(self.algos)
        summary = []
        for i, algo in enumerate(self.algos):
            p = self.probs[algo][rows[algo]]
            xs = x + (i - (len(self.algos) - 1) / 2) * h
            ax.bar(xs, p[shown], width=h * 0.92, color=COLORS[algo], label=algo)
            for xi, v in zip(xs, p[shown]):
                if v >= 0.05:
                    ax.text(xi, v + 0.015, f'{v:.2f}', ha='center', fontsize=8, color=TEXT_2)
            summary.append(f'{algo} -> {self.classes[p.argmax()]} ({p.max():.0%})')
        ax.set_xticks(x, [self.classes[j] for j in shown], rotation=45 if len(shown) <= 20 else 90, ha='right')
        ax.get_xticklabels()[shown.index(label)].set_fontweight('bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='x', visible=False)
        ax.set_ylabel('predicted probability')
        ax.set_title('   '.join(summary) + '   (bold = true class)', fontsize=10, pad=24)
        _top_legend(ax)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ feature space
    def _class_ids(self, classes):
        classes = [classes] if isinstance(classes, (str, int)) else classes
        return [self.classes.index(c) if isinstance(c, str) else int(c) for c in classes]

    def pred_composition(self, cls, domain=None):
        """True-label breakdown of the samples each method predicts as `cls` (what gets absorbed into it)."""
        cid = self._class_ids(cls)[0]
        out = {}
        for algo in self.algos:
            d = self._select(algo, domain)
            out[algo] = d.label[d.pred == cid].map(lambda c: self.classes[c]).value_counts()
        table = pd.DataFrame(out).fillna(0).astype(int)
        return table.sort_values(self.algos[-1], ascending=False)

    def select_samples(self, classes, select='pred', algo='Ramen', domains=None, max_points=3000, seed=None):
        """
        Samples for a feature-space plot, one row per test image (Ramen-based columns + `_base` NoAdapt ones).
        classes: class names or indices.
        select:  'pred'   -> `algo` predicted one of `classes`
                 'true'   -> true label is one of `classes`
                 'either' -> union of the two
                 'wrong'  -> `algo` predicted one of `classes` but was wrong (what gets absorbed)
        domains: corruption name(s) to keep, or None for all.
        """
        ids = self._class_ids(classes)
        has_both = {'NoAdapt', 'Ramen'} <= set(self.results)
        d = self.paired() if has_both else self.results[algo].copy()
        pred = d['pred_base' if (has_both and algo == 'NoAdapt') else 'pred']
        correct = d['correct_base' if (has_both and algo == 'NoAdapt') else 'correct']
        in_pred, in_true = pred.isin(ids), d.label.isin(ids)
        mask = {'pred': in_pred, 'true': in_true, 'either': in_pred | in_true, 'wrong': in_pred & ~correct}[select]
        if domains is not None:
            domains = [domains] if isinstance(domains, str) else domains
            mask &= d.domain_idx.isin([self.environments.index(x) for x in domains])
        d = d[mask].copy()
        d['sel_pred'], d['sel_correct'] = pred[mask], correct[mask]
        if len(d) > max_points:
            d = d.sample(max_points, random_state=self.seed if seed is None else seed)
        d['true_name'] = [self.classes[c] for c in d.label]
        d['pred_name'] = [self.classes[c] for c in d.sel_pred]
        d.attrs.update(classes=[self.classes[i] for i in ids], select=select, algo=algo)
        return d

    def _features_for(self, d, features_from):
        if features_from not in self.features:
            raise RuntimeError(f'No features for {features_from}; run exp.evaluate(capture_features=True).')
        src = self.results[features_from]
        n = max(len(ds) for ds in self.datasets.datasets)  # row stride; domains may differ in size
        lookup = np.full(len(self.environments) * n, -1)
        lookup[src.domain_idx.values * n + src.sample_idx.values] = np.arange(len(src))
        rows = lookup[d.domain_idx.values * n + d.sample_idx.values]
        return self.features[features_from][rows].astype(np.float32)

    def tsne(self, d, features_from='Ramen', perplexity=30, pca_dim=50, seed=None):
        """2-D t-SNE of the selected samples' features; returns a copy of `d` with columns x, y."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        X = self._features_for(d, features_from)
        if X.shape[1] > pca_dim and len(X) > pca_dim:
            X = PCA(pca_dim, random_state=0).fit_transform(X)  # standard denoising step before t-SNE
        seed = self.seed if seed is None else seed
        xy = TSNE(2, perplexity=min(perplexity, (len(X) - 1) / 3), init='pca', learning_rate='auto',
                  random_state=seed).fit_transform(X)
        return d.assign(x=xy[:, 0], y=xy[:, 1])

    def _color_map(self, d, color_by, max_colors=8):
        """Category per row + {category: colour}; beyond 8 categories the rest fold into 'other'."""
        if color_by == 'outcome':
            fixed = {'fixed': BLUE, 'broken': RED, 'both right': '#c3c2b7', 'both wrong': TEXT_2}
            return d['outcome'], {k: v for k, v in fixed.items() if k in set(d['outcome'])}
        if color_by == 'correct':
            cats = np.where(d.sel_correct, 'correct', 'wrong')
            return pd.Series(cats, index=d.index), {'correct': '#c3c2b7', 'wrong': RED}
        col = {'label': 'true_name', 'pred': 'pred_name', 'domain': 'domain'}[color_by]
        cats = d[col].astype(str)
        top = cats.value_counts().index
        if len(top) > max_colors:
            keep = set(top[:max_colors - 1])
            cats = cats.where(cats.isin(keep), 'other')
            top = list(top[:max_colors - 1]) + ['other']
        cmap = {c: (MUTED if c == 'other' else CATEGORICAL[i]) for i, c in enumerate(top)}
        return cats, cmap

    def plot_tsne(self, classes, select='pred', algo='Ramen', features_from=None, color_by='label',
                  domains=None, max_points=3000, perplexity=30, seed=None, point_size=14):
        """
        t-SNE of image features for samples tied to `classes` (see select_samples for `select`).
        features_from: method(s) whose features to embed, e.g. 'Ramen', 'NoAdapt' or ('NoAdapt', 'Ramen')
                       for a side-by-side view of how adaptation moves the same samples. Default: all captured.
        color_by: 'label' (true class) | 'pred' | 'domain' | 'outcome' (fixed/broken vs NoAdapt) | 'correct'.
        Marker: o = `algo` got it right, x = wrong.
        Returns (fig, list of per-panel DataFrames with x, y).
        """
        sources = features_from or [a for a in self.algos if a in self.features]
        sources = [sources] if isinstance(sources, str) else list(sources)
        d = self.select_samples(classes, select, algo, domains, max_points, seed)
        if len(d) < 5:
            raise ValueError(f'Only {len(d)} samples match; widen classes/select/domains.')
        cats, cmap = self._color_map(d, color_by)

        fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources) + 2.4, 6.2), squeeze=False)
        panels = []
        for ax, src in zip(axes[0], sources):
            e = self.tsne(d, src, perplexity=perplexity, seed=seed)
            panels.append(e)
            for cat, color in cmap.items():  # draw in legend order; 'other' first so it sits underneath
                for ok, marker in ((True, 'o'), (False, 'X')):
                    m = (cats.values == cat) & (e.sel_correct.values == ok)
                    ax.scatter(e.x[m], e.y[m], s=point_size, marker=marker, color=color, alpha=0.75,
                               linewidths=0, zorder=1 if cat == 'other' else 2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f'{src} features' + (' (after adaptation)' if src == 'Ramen' else ' (zero-shot)'))

        counts = cats.value_counts()
        handles = [plt.Line2D([], [], ls='', marker='o', ms=7, color=c, label=f'{k}  ({counts.get(k, 0)})')
                   for k, c in cmap.items()]
        handles += [plt.Line2D([], [], ls='', marker='o', ms=7, color=TEXT_2, label=f'{algo} correct'),
                    plt.Line2D([], [], ls='', marker='X', ms=7, color=TEXT_2, label=f'{algo} wrong')]
        legend_w = 2.4 / fig.get_figwidth()  # fraction of the figure kept free for the legend
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1 - legend_w, 0.5),
                   title=color_by, title_fontsize=9, alignment='left')
        names = ', '.join(d.attrs['classes'])
        what = {'pred': f'predicted as {names} by {algo}',
                'true': f'whose true class is {names}',
                'either': f'predicted as {names} by {algo}, or truly {names}',
                'wrong': f'wrongly predicted as {names} by {algo}'}[select]
        where = f' | {", ".join([domains] if isinstance(domains, str) else domains)}' if domains else ''
        fig.suptitle(f'{self.dataset_name}: t-SNE of samples {what}{where}  (n={len(d)})',
                     x=0.01, ha='left', fontsize=12, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1 - legend_w, 1))
        return fig, panels

    def show_failures(self, domain=None, kind='broken', n=16, cols=8):
        """
        Grid of test images, most confident first.
        kind: 'broken' (NoAdapt right, Ramen wrong) | 'both wrong' | 'fixed' | 'wrong' (any Ramen error)
        """
        if kind == 'wrong':
            d = self.results['Ramen']
            d = d[~d.correct]
        else:
            d = self.paired()
            d = d[d.outcome == kind]
        if domain is not None:
            d = d[d.domain_idx == self.environments.index(domain)]
        d = d.sort_values('conf', ascending=False).head(n)
        if d.empty:
            print(f'No "{kind}" samples for {domain or f"any {self.env_label}"}.')
            return None

        rows = int(np.ceil(len(d) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 2.2), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, r in zip(axes.flat, d.itertuples()):
            ax.imshow(self._image(r.domain_idx, r.sample_idx), interpolation='nearest')
            title = f'{self.classes[r.label]}\n-> {self.classes[r.pred]} {r.conf:.0%}'
            if domain is None:
                title = f'{self.environments[r.domain_idx]}\n' + title
            ax.set_title(title, fontsize=7, color=TEXT, loc='center', fontweight='normal')
        fig.suptitle(f'{self.dataset_name} | {kind} | {domain or self._all_envs} '
                     f'(true -> Ramen pred, confidence)', x=0.01, ha='left', fontsize=11, fontweight='bold')
        fig.tight_layout()
        return fig

    def show_confused_pairs(self, pairs=None, domain=None, top_k=4, n=8, algo='Ramen', order='confident', seed=0):
        """
        One row of test images per confused pair (A -> B): samples whose TRUE class is A and that `algo` predicts as B.
        pairs=None uses the top_k pairs from confused_pairs(). order: 'confident' (most confidently wrong first),
        'borderline' (least confident first) or 'random'. Each caption gives the corruption, `algo`'s confidence in B
        and, for every other method, its verdict on the same image: 'right' (predicted A), 'same error' (also B)
        or the other wrong class it picked.
        """
        if pairs is None:
            pairs = list(self.confused_pairs(domain, top_k).index)
        d = self._select(algo, domain)
        keys = ['domain_idx', 'sample_idx']
        others = [a for a in self.algos if a != algo and a in self.results]
        for o in others:
            d = d.merge(self.results[o][keys + ['pred']].rename(columns={'pred': f'pred_{o}'}), on=keys, how='left')

        fig, axes = plt.subplots(len(pairs), n + 1, figsize=(1.45 * n + 1.6, 2.25 * len(pairs)), squeeze=False,
                                 gridspec_kw={'width_ratios': [1.1] + [1] * n})
        for ax in axes.flat:
            ax.axis('off')
        for row, (a, b) in zip(axes, pairs):
            ia, ib = self._class_ids([a, b])
            m = d[(d.label == ia) & (d.pred == ib)]
            share = len(m) / max((d.label == ia).sum(), 1) * 100
            row[0].text(0.95, 0.5, f'true: {self.classes[ia]}\n{algo}: {self.classes[ib]}\n\n{len(m)} samples\n'
                                   f'({share:.1f}% of {self.classes[ia]})',
                        transform=row[0].transAxes, ha='right', va='center', fontsize=9, color=TEXT)
            if order == 'random':
                m = m.sample(min(n, len(m)), random_state=seed)
            else:
                m = m.sort_values('conf', ascending=(order == 'borderline')).head(n)
            for ax, r in zip(row[1:], m.itertuples()):
                ax.imshow(self._image(r.domain_idx, r.sample_idx), interpolation='nearest')
                title = [f'{algo} {r.conf:.0%}']
                for o in others:
                    po = int(getattr(r, f'pred_{o}'))
                    title.append(f'{o}: ' + ('right' if po == ia else 'same error' if po == ib else self.classes[po]))
                if domain is None:
                    title.insert(0, self.environments[r.domain_idx])
                ax.set_title('\n'.join(title), fontsize=7, color=TEXT, loc='center', fontweight='normal')
        fig.suptitle(f'{self.dataset_name}: most confused pairs, {domain or self._all_envs} '
                     f'(images of the true class that {algo} predicts as the other; {order} first)',
                     x=0.01, ha='left', fontsize=11, fontweight='bold')
        fig.tight_layout()
        return fig


class CIFAR10CExperiment(RamenExperiment):
    dataset_name = 'CIFAR10C'
    default_model = 'clip_vitbase32'
    watch_classes = ('deer', 'truck', 'frog')
    superclasses = {c: 'vehicle' if c in ('airplane', 'automobile', 'ship', 'truck') else 'animal'
                    for c in ['airplane', 'automobile', 'bird', 'cat', 'deer',
                              'dog', 'frog', 'horse', 'ship', 'truck']}


_CIFAR100_SUPERCLASSES = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit & vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omni/herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


class _ManyClassExperiment(RamenExperiment):
    """Datasets with too many classes for a full confusion matrix."""

    def plot_confusion(self, algo='Ramen', domain=None, top_k=25):
        """KxK is unreadable, so show the most frequent (true -> predicted) error pairs instead."""
        d = self._select(algo, domain)
        err = d[~d.correct]
        per_class = d.groupby('label').size()
        pairs = err.groupby(['label', 'pred']).size().nlargest(top_k)
        pct = pairs / per_class.reindex(pairs.index.get_level_values(0)).values * 100
        if self.superclasses is None:
            within = np.zeros(len(pairs), dtype=bool)
        else:
            within = np.array([self.superclasses[self.classes[t]] == self.superclasses[self.classes[p]]
                               for t, p in pairs.index])
        labels = [f'{self.classes[t]} -> {self.classes[p]}' for t, p in pairs.index]
        y = np.arange(len(pairs))

        fig, ax = plt.subplots(figsize=(8.5, 0.3 * len(pairs) + 1.4))
        if self.superclasses is None:
            ax.barh(y, pairs.values, height=0.65, color=BLUE)
        else:
            ax.barh(y[within], pairs.values[within], height=0.65, color=AQUA, label='same superclass (fine-grained)')
            ax.barh(y[~within], pairs.values[~within], height=0.65, color=MUTED, label='different superclass')
        for yi, c, v in zip(y, pairs.values, pct.values):
            ax.text(c + pairs.max() * 0.01, yi, f'{c}  ({v:.0f}% of class)', va='center', fontsize=8, color=TEXT_2)
        ax.set_xlim(right=pairs.max() * 1.3)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('number of errors (true -> predicted)')
        ax.set_title(f'{algo}: top {top_k} confusions on {domain or self._all_envs}',
                     pad=24 if self.superclasses is not None else 6)
        if self.superclasses is not None:
            _top_legend(ax)
        fig.tight_layout()
        return fig


class CIFAR100CExperiment(_ManyClassExperiment):
    dataset_name = 'CIFAR100C'
    default_model = 'clip_vitbase16'
    watch_classes = ('tractor', 'oak_tree', 'possum')
    superclasses = {c: g for g, cs in _CIFAR100_SUPERCLASSES.items() for c in cs}
    heatmap_by_superclass = True


assert len(CIFAR100CExperiment.superclasses) == 100


class _FileImageExperiment(_ManyClassExperiment):
    """Datasets whose domains are image folders on disk (`.samples` = [(path, label)]) instead of arrays."""

    @staticmethod
    def _keep_samples(ds, idx):
        ds.samples = [ds.samples[i] for i in idx]
        ds.targets = [t for _, t in ds.samples]
        if hasattr(ds, 'imgs'):
            ds.imgs = ds.samples

    def _image(self, domain_idx, sample_idx, size=224):
        path = self.datasets[domain_idx].samples[sample_idx][0]
        with Image.open(path) as img:
            img = img.convert('RGB')
            img.thumbnail((size, size))
            return np.asarray(img)


class ImageNetC5KExperiment(_FileImageExperiment):
    """
    ImageNet-C, severity 5: 15 corruptions of the same 5,000 validation images (src/datasets/imagenet_test_image_ids.txt),
    1000 classes. Images are read from disk. n_per_domain: the same random subset of images for every corruption.
    """
    dataset_name = 'ImageNetC5K'
    default_model = 'clip_vitbase16'

    def _build_datasets(self, preprocess):
        datasets = get_dataset_class(self.dataset_name)(root=self.data_root, severity=self.severity,
                                                        transform=preprocess)
        if self.n_per_domain:
            # Every corruption lists the same image ids in the same (sorted) order
            n = len(datasets[0])
            idx = np.sort(np.random.default_rng(self.seed).choice(n, self.n_per_domain, replace=False))
            for ds in datasets.datasets:
                self._keep_samples(ds, idx)
        return datasets


class DomainNetExperiment(_FileImageExperiment):
    """
    DomainNet (6 domains x 345 classes, ~596k images). Unlike the corruption datasets, every domain holds
    different images and has its own size.
    n_per_domain: random subset of that many images per domain (drawn independently per domain).
    """
    dataset_name = 'DomainNet'
    default_model = 'clip_vitbase32'
    env_label = 'domain'

    def _build_datasets(self, preprocess):
        datasets = get_dataset_class(self.dataset_name)(root=self.data_root, transform=preprocess)
        if self.n_per_domain:
            rng = np.random.default_rng(self.seed)
            for ds in datasets.datasets:  # torchvision ImageFolder
                self._keep_samples(ds, np.sort(rng.choice(len(ds), min(self.n_per_domain, len(ds)), replace=False)))
        return datasets


class EuroSATExperiment(DomainNetExperiment):
    """
    EuroSAT RGB (27,000 64x64 Sentinel-2 images, 10 land-cover classes). A single domain, so the shift is
    natural photos -> satellite images rather than a mix of domains; 'mixed' is one shuffled stream.
    Prompts come from cfg/EuroSAT (a satellite-photo template, not the default ensemble).
    """
    dataset_name = 'EuroSAT'
    default_model = 'clip_vitbase16'
    superclasses = {**{c: 'vegetation' for c in ['Annual Crop Land', 'Forest', 'Herbaceous Vegetation Land',
                                                 'Pasture Land', 'Permanent Crop Land']},
                    **{c: 'built' for c in ['Highway or Road', 'Industrial Buildings', 'Residential Buildings']},
                    **{c: 'water' for c in ['River', 'Sea or Lake']}}
    plot_confusion = RamenExperiment.plot_confusion  # 10 classes: the full matrix is readable


EXPERIMENTS = {'CIFAR10C': CIFAR10CExperiment, 'CIFAR100C': CIFAR100CExperiment,
               'ImageNetC5K': ImageNetC5KExperiment, 'DomainNet': DomainNetExperiment,
               'EuroSAT': EuroSATExperiment}


def make_experiment(dataset, **kwargs):
    """make_experiment('CIFAR10C', tta_mode='mixed') -> CIFAR10CExperiment(...)"""
    return EXPERIMENTS[dataset](**kwargs)
