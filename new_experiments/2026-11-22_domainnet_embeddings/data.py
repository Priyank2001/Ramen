"""
Zero-shot CLIP embeddings of DomainNet, read from cached runs (offline, CPU, read-only).

Every array is row-aligned with the seed-0 NoAdapt run (stream order). Nothing is recomputed on the GPU.

    d = DomainNetEmbeddings()
    d.Z (N, 512) float32 unit-norm image features      d.W (345, 512) float32 unit-norm class text embeddings
    d.y, d.dom, d.pred (N,) true label / domain id / zero-shot prediction
    d.describe()                                        images, classes and zero-shot errors per domain
    d.image(i)                                          PIL image of row i (read from the HDD)
"""
import os

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, make_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
RESULTS = os.path.join(NEW, 'results')
# Text embeddings cached by the hubness experiment: encode_text(CLIP ViT-B/32, class folder names, Ramen's templates)
TEXT = os.path.join(NEW, '2026-11-21_hubness', 'results', 'DomainNet_text_features.npy')
MODEL = 'clip_vitbase32'
LOGIT_SCALE = 100.0  # CLIP's logit scale, as in Ramen / NoAdapt


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class DomainNetEmbeddings:
    """Cached zero-shot CLIP features, text embeddings, labels and domains of all 596,006 DomainNet images."""

    def __init__(self, seed=0, data_root='~/data'):
        prefix = os.path.join(RESULTS, f'DomainNet_{MODEL}_mixed_nall_bs100_s{seed}_NoAdapt')
        self.df = pd.read_pickle(prefix + '.pkl')
        assert (self.df.position.values == np.arange(len(self.df))).all()
        self.Z = _unit(np.load(prefix + '_feats.npy').astype(np.float32))
        self.W = _unit(np.load(TEXT).astype(np.float32))
        assert len(self.Z) == len(self.df), (self.Z.shape, len(self.df))

        self.root = os.path.join(os.path.expanduser(data_root), 'domainbed', 'domain_net')
        names = self.df.groupby('domain_idx', observed=True).domain.first()
        self.domains = [str(names[d]) for d in range(len(names))]
        # ImageFolder's class order = sorted folder names; Ramen's text embeddings use the same order
        self.classes = sorted(e.name for e in os.scandir(os.path.join(self.root, self.domains[0])) if e.is_dir())
        assert len(self.classes) == len(self.W)

        self.y = self.df.label.values.astype(np.int64)
        self.dom = self.df.domain_idx.values.astype(np.int64)
        self.pred = self.df.pred.values.astype(np.int64)
        self.correct = self.df.correct.values.astype(bool)
        self.sample_idx = self.df.sample_idx.values.astype(np.int64)
        self.C, self.n_dom = len(self.classes), len(self.domains)
        self._files = {}  # (domain, class) -> sorted file list, filled on demand

    def __len__(self):
        return len(self.y)

    # ------------------------------------------------------------------ tables
    def describe(self):
        """Images, classes present and zero-shot errors per domain (counts with percentages)."""
        rows = []
        for d, name in enumerate(self.domains):
            m = self.dom == d
            n, wrong = int(m.sum()), int((~self.correct[m]).sum())
            per_class = np.bincount(self.y[m], minlength=self.C)
            rows.append(dict(domain=name, images=n, share=f'{100 * n / len(self):.1f}%',
                             classes_present=int((per_class > 0).sum()),
                             median_per_class=int(np.median(per_class[per_class > 0])),
                             zero_shot_errors=f'{wrong:,} of {n:,} ({100 * wrong / n:.1f}%)',
                             zero_shot_acc=round(100 * (n - wrong) / n, 2)))
        out = pd.DataFrame(rows).set_index('domain')
        print(f'domain-averaged zero-shot accuracy {out.zero_shot_acc.mean():.2f}, '
              f'sample-weighted {100 * self.correct.mean():.2f} ({self.correct.sum():,} of {len(self):,})')
        return out

    def sanity(self):
        """Recompute the zero-shot prediction from the cached features and compare with the cached NoAdapt run."""
        pred = np.concatenate([np.argmax(self.Z[s:s + 65536] @ self.W.T, 1) for s in range(0, len(self), 65536)])
        same = int((pred == self.pred).sum())
        print(f'argmax(z·W) == cached NoAdapt prediction on {same:,} of {len(self):,} images '
              f'({100 * same / len(self):.2f}%; differences come from the fp16 feature cache)')
        return same / len(self)

    # ------------------------------------------------------------------ images
    def _class_files(self, d, c):
        key = (d, c)
        if key not in self._files:
            # Same listing ImageFolder builds, restricted to one class folder
            files = make_dataset(os.path.join(self.root, self.domains[d]), {self.classes[c]: c},
                                 extensions=IMG_EXTENSIONS)
            self._files[key] = [f for f, _ in files]
        return self._files[key]

    def image_path(self, i):
        """File of row i. sample_idx is the index into ImageFolder(domain).samples, which is sorted by class."""
        d, c, s = self.dom[i], self.y[i], self.sample_idx[i]
        counts = np.bincount(self.y[self.dom == d], minlength=self.C)
        offset = int(counts[:c].sum())
        files = self._class_files(d, c)
        assert len(files) == counts[c], (self.domains[d], self.classes[c], len(files), counts[c])
        return files[s - offset]

    def image(self, i):
        return Image.open(self.image_path(i)).convert('RGB')
