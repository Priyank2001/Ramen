"""
The recorded runs the sink case study reads: Ramen and the true-label ORACLE on the same test stream, each with
per-image probability vectors and CLIP embeddings before / after adaptation (src/methods/RamenRecorder.py, run by
2026-10-28_confusion_records/run_records.py), plus the CLIP text features so logits can be recomputed.

    runs = SinkRuns('DomainNet', seed=0)
    runs.sanity_check()              # recorded runs == their plain cached runs, streams aligned, logits reproduce
    runs.table                       # one row per image, stream order: label, zs_pred, ramen_pred, oracle_pred, ...
    runs.logits('ramen', rows)       # 100 * embedding @ text_features.T, the logits Ramen classified with
    runs.probs('oracle', rows)       # the recorded softmax
    runs.image(row)                  # the raw test image

Logits are exact up to fp16 rounding: CLIP's classifier is `100 * feature @ text_feature.T` with fixed text
features (src/models/ModelForBySampleTTA.py), and only the image encoder's norm layers are adapted.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS_DIR = os.path.join(os.path.dirname(HERE), '2026-10-28_confusion_records')
sys.path.insert(0, os.path.dirname(HERE))  # new_experiments/, for ramen_analysis
sys.path.insert(0, RECORDS_DIR)  # RecordExperiment / ImageRecords (read-only import; it also puts src/ on the path)

from experiment import RecordExperiment, count_pct  # noqa: E402

# prediction source -> (recorded run, embedding the logits come from, recorded probability vector)
SOURCES = {
    'zero_shot': ('Ramen', 'emb_before', 'zs_probs'),   # the pass that files the image in Ramen's cache
    'ramen': ('Ramen', 'emb_after', 'probs'),
    'oracle': ('true_label', 'emb_after', 'probs'),     # ORACLE: its cache is filed by the true label
}
SOURCE_NAMES = {'zero_shot': 'zero-shot', 'ramen': 'Ramen', 'oracle': 'oracle cache'}
# plain cached run each recorded run must reproduce (same stream, same seed)
REFERENCES = {'Ramen': 'Ramen', 'true_label': 'oracle_file_true'}


class TextFeatures:
    """The CLIP class embeddings Ramen classifies against, computed once on the GPU (fp16, as in the runs)."""

    def __init__(self, exp, classes):
        self.exp, self.classes = exp, classes
        self.path = exp._cache_path('pillow_sink_text_features')[:-len('.pkl')] + '.npy'

    def load(self, device='cuda'):
        if not os.path.exists(self.path):
            np.save(self.path, self._compute(device))
        return np.load(self.path).astype(np.float32)

    def _compute(self, device):
        import torch
        from models.clip_utils import DEFAULT_TEMPLATES, encode_text
        from models.get_pretrained_model import get_pretrained_model
        model, _ = get_pretrained_model(SimpleNamespace(model=self.exp.model_name, device=device))
        templates = self.exp._load_cfg('Ramen').get('templates') or DEFAULT_TEMPLATES
        with torch.no_grad():
            t = encode_text(model, self.classes, templates)
        del model
        return t.cpu().numpy().astype(np.float16)


class SinkRuns:
    """Ramen and the true-label oracle, recorded on one dataset and seed, row-aligned in stream order."""

    def __init__(self, dataset='DomainNet', seed=0, device='cuda'):
        self.dataset, self.seed = dataset, seed
        self.recorded = {v: RecordExperiment(dataset, seed=seed, variant=v) for v in ('Ramen', 'true_label')}
        for v, r in self.recorded.items():
            if not r.is_cached():
                raise RuntimeError(f'{v} is not recorded for {dataset} seed {seed}: run '
                                   f'2026-10-28_confusion_records/run_records.py {dataset} {seed} --variant {v}')
        self.records = {v: r.records for v, r in self.recorded.items()}
        self.ramen_exp = self.recorded['Ramen']
        self.classes = self.ramen_exp.classes
        self.exp = self.ramen_exp.exp
        self.cfg = self.exp._load_cfg('Ramen')
        self.text = TextFeatures(self.exp, self.classes).load(device)
        self.table = self._table()
        self._entropy = None

    def _table(self):
        r, o = self.records['Ramen'].table, self.records['true_label'].table
        t = r[['position', 'domain', 'domain_idx', 'sample_idx', 'label', 'true_class', 'zs_pred', 'zs_class',
               'zs_conf', 'zs_true_prob']].copy()
        t['ramen_pred'], t['ramen_class'], t['ramen_conf'] = r['pred'], r['pred_class'], r['conf']
        t['oracle_pred'], t['oracle_class'], t['oracle_conf'] = o['pred'], o['pred_class'], o['conf']
        return t

    # ------------------------------------------------------------------ vectors
    @property
    def num_classes(self):
        return len(self.classes)

    def embeddings(self, source, rows=None):
        run, emb, _ = SOURCES[source]
        a = self.records[run].arrays[emb]
        return np.asarray(a if rows is None else a[rows], dtype=np.float32)

    def logits(self, source, rows):
        return 100.0 * self.embeddings(source, rows) @ self.text.T

    def probs(self, source, rows):
        run, _, p = SOURCES[source]
        return np.asarray(self.records[run].arrays[p][rows], dtype=np.float32)

    def zero_shot_entropy(self, chunk=50_000):
        """Entropy of each image's zero-shot prediction: what Ramen stores with its cache entry."""
        if self._entropy is None:
            n = len(self.table)
            out = np.empty(n, dtype=np.float32)
            for s in range(0, n, chunk):
                z = self.logits('zero_shot', slice(s, s + chunk))
                z -= z.max(1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(1, keepdims=True)
                out[s:s + chunk] = -(p * np.log(np.clip(p, 1e-30, None))).sum(1)
            self._entropy = out
        return self._entropy

    def image(self, row):
        r = self.table.iloc[row]
        return self.ramen_exp.image(int(r.domain_idx), int(r.sample_idx))

    @property
    def environments(self):
        return list(self.ramen_exp._datasets().environments)

    # ------------------------------------------------------------------ tables
    def data_table(self):
        """Images and classes per domain: the denominators of everything else."""
        g = self.table.groupby('domain', observed=True)
        out = pd.DataFrame({'images': g.size(), 'classes present': g['label'].nunique()})
        out.loc['all'] = [len(self.table), self.table['label'].nunique()]
        return out.astype(int)

    def accuracy_table(self):
        """Correct images and accuracy per domain under zero-shot, Ramen and the oracle; last row = domain average."""
        t = self.table
        out = pd.DataFrame({'images': t.groupby('domain', observed=True).size()})
        for col, name in [('zs_pred', 'zero-shot'), ('ramen_pred', 'Ramen'), ('oracle_pred', 'oracle cache')]:
            ok = (t[col] == t['label']).groupby(t['domain'], observed=True)
            out[f'{name} correct'] = ok.sum()
            out[f'{name} %'] = ok.mean() * 100
        avg = out.mean()
        avg['images'] = len(t)
        for name in ('zero-shot', 'Ramen', 'oracle cache'):
            avg[f'{name} correct'] = out[f'{name} correct'].sum()
        out.loc['domain avg'] = avg
        return out.round(2).astype({c: int for c in out.columns if not c.endswith('%')})

    # ------------------------------------------------------------------ checks
    def sanity_check(self, n_logit_rows=20_000, seed=0):
        """
        1. each recorded run predicts exactly what its plain cached run predicts (same stream): must be 100.00%
        2. both recorded runs see the same stream, and the same zero-shot embeddings (adaptation starts from them)
        3. recomputed zero-shot logits give the recorded zero-shot class; their softmax matches the recorded one
        Returns a table; raises if a must-be-exact check fails.
        """
        rows, ok = [], True
        keys = ['domain_idx', 'sample_idx']
        for v, ref in REFERENCES.items():
            cached = pd.read_pickle(self.exp._cache_path(ref))
            rec = self.records[v].table
            aligned = (cached[keys].to_numpy() == rec[keys].to_numpy()).all()
            same = int((cached['pred'].to_numpy() == rec['pred'].to_numpy()).sum()) if aligned else 0
            rows.append((f'recorded {v} prediction == cached {ref}', count_pct(same, len(rec))))
            ok &= aligned and same == len(rec)
        r, o = self.records['Ramen'], self.records['true_label']
        aligned = (r.table[keys].to_numpy() == o.table[keys].to_numpy()).all()
        rows.append(('Ramen and oracle runs see the same stream', 'yes' if aligned else 'NO'))
        ok &= aligned
        sample = np.sort(np.random.default_rng(seed).choice(len(self.table), n_logit_rows, replace=False))
        diff = np.abs(r.arrays['emb_before'][sample].astype(np.float32) -
                      o.arrays['emb_before'][sample].astype(np.float32)).max()
        rows.append((f'zero-shot embedding, Ramen vs oracle run (max |diff|, {n_logit_rows:,} images)', f'{diff:.2e}'))
        z = self.logits('zero_shot', sample)
        same = int((z.argmax(1) == self.table['zs_pred'].to_numpy()[sample]).sum())
        rows.append(('argmax of recomputed zero-shot logits == recorded zero-shot class', count_pct(same, len(sample))))
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        err = np.abs(p - self.probs('zero_shot', sample)).max(1)
        rows.append(('softmax(recomputed logits) vs recorded zero-shot probs: median / max |diff|',
                     f'{np.median(err):.1e} / {err.max():.1e}'))
        table = pd.DataFrame(rows, columns=['check', 'result']).set_index('check')
        if not ok:
            raise AssertionError(f'sanity check failed:\n{table.to_string()}')
        return table
