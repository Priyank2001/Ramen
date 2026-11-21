import os
from torchvision.datasets import ImageFolder

from .corruption.utils import read_classnames


class ImageNetVariants:
    """
    ImageNet-R (30,000 renditions, 200 classes), ImageNet-A (7,500 natural adversarial photos, 200 classes) and
    ImageNet-Sketch (50,889 sketches, 1,000 classes) as three domains over the full ImageNet-1k label space.
    Folders are WordNet ids; every domain's labels are remapped to the 1,000-class index (sorted wnids, the order of
    ImageNet-C's classnames.txt), so one classifier covers the mixed stream.
    Expects root/imagenet_variants/{imagenet-r,imagenet-a,sketch}/<wnid>/* and root/corruption/ImageNet-C/classnames.txt.
    """
    FOLDERS = {'rendition': 'imagenet-r', 'adversarial': 'imagenet-a', 'sketch': 'sketch'}

    def __init__(self, root, transform=None):
        classnames = read_classnames(os.path.join(root, 'corruption', 'ImageNet-C', 'classnames.txt'))
        wnids = sorted(classnames)
        index = {w: i for i, w in enumerate(wnids)}

        self.environments = list(self.FOLDERS)
        self.datasets = []
        for env in self.environments:
            ds = ImageFolder(os.path.join(root, 'imagenet_variants', self.FOLDERS[env]), transform=transform)
            # ImageFolder numbers only the folders present (200 for R / A); map them to the global ImageNet index
            ds.samples = [(path, index[ds.classes[t]]) for path, t in ds.samples]
            ds.imgs = ds.samples
            ds.targets = [t for _, t in ds.samples]
            ds.classes, ds.class_to_idx = wnids, index
            self.datasets.append(ds)

        self.num_classes = len(wnids)
        self.classes = [classnames[w] for w in wnids]

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
