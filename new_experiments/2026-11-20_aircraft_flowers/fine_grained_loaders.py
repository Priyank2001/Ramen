"""FGVC-Aircraft and Flowers102 loaders for the Ramen harness (src/ is read-only, so they live in this folder)."""
import json
import os

import scipy.io
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
HERE = os.path.dirname(os.path.abspath(__file__))


class ListedImages(Dataset):
    """Images on disk as the harness expects them: `.samples` = [(path, label)], `.targets`, items are (image, label)."""

    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert('RGB')
        return (self.transform(image) if self.transform else image), label


class _SingleDomain:
    """One dataset, one domain: the interface the harness reads (environments, datasets, classes, num_classes)."""
    environment = None

    def __init__(self, samples, classes, transform):
        self.environments = [self.environment]
        self.datasets = [ListedImages(samples, transform)]
        self.classes = classes
        self.num_classes = len(classes)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Aircraft(_SingleDomain):
    """
    FGVC-Aircraft 2013b, 100 `variant` classes (e.g. "707-320", "A300B4"); test split = 3,333 images.
    Expects root/fgvc-aircraft-2013b/data/{images/*.jpg, variants.txt, images_variant_<split>.txt}.
    """
    environment = 'aircraft'

    def __init__(self, root, transform=None, split='test'):
        assert split in ('train', 'val', 'trainval', 'test'), split
        data = os.path.join(root, 'fgvc-aircraft-2013b', 'data')
        with open(os.path.join(data, 'variants.txt')) as f:
            classes = [line.strip() for line in f if line.strip()]
        index = {name: i for i, name in enumerate(classes)}
        samples = []
        with open(os.path.join(data, f'images_variant_{split}.txt')) as f:
            for line in f:
                image_id, variant = line.strip().split(' ', 1)  # variant names can contain spaces
                samples.append((os.path.join(data, 'images', image_id + '.jpg'), index[variant]))
        super().__init__(samples, classes, transform)


class Flowers102(_SingleDomain):
    """
    Oxford Flowers102, 102 classes; test split (setid.mat `tstid`) = 6,149 images.
    Expects root/flowers-102/{jpg/image_#####.jpg, imagelabels.mat, setid.mat}. Class names (CoOp / CLIP's) are in
    flowers102_classnames.json next to this file, ordered by the dataset's label 1..102.
    """
    environment = 'flowers'
    SPLIT_KEYS = {'train': 'trnid', 'val': 'valid', 'test': 'tstid'}

    def __init__(self, root, transform=None, split='test'):
        base = os.path.join(root, 'flowers-102')
        ids = scipy.io.loadmat(os.path.join(base, 'setid.mat'), squeeze_me=True)[self.SPLIT_KEYS[split]]
        labels = scipy.io.loadmat(os.path.join(base, 'imagelabels.mat'), squeeze_me=True)['labels']  # 1-based, per image id
        with open(os.path.join(HERE, 'flowers102_classnames.json')) as f:
            classes = json.load(f)
        samples = [(os.path.join(base, 'jpg', f'image_{i:05d}.jpg'), int(labels[i - 1]) - 1) for i in ids]
        super().__init__(samples, classes, transform)
