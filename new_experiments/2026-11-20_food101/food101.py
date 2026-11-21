"""Food-101 loader for the Ramen harness (a local copy of the src/datasets/eurosat.py pattern; src/ is read-only)."""
import os

from torchvision.datasets import ImageFolder


class Food101:
    """
    Food-101 (101 food categories, 1,000 photos each; official split 750 train / 250 test per class), one domain.
    Expects root/food-101/{images/<class>/<id>.jpg, meta/{train,test}.txt, meta/classes.txt} (the official
    ETH archive, md5 85eeb15f3717b99a5da872d97d918f87). Default split 'test' = 25,250 images, the TTA convention.
    Class order = sorted folder names = meta/classes.txt, so labels match the CLIP prompts built from `classes`.
    """

    def __init__(self, root, transform=None, split='test'):
        assert split in ('train', 'test'), split
        base = os.path.join(root, 'food-101')
        with open(os.path.join(base, 'meta', f'{split}.txt')) as f:
            wanted = {os.path.join(base, 'images', line.strip() + '.jpg') for line in f if line.strip()}

        self.environments = ['food']
        self.datasets = [ImageFolder(os.path.join(base, 'images'), transform=transform,
                                     is_valid_file=lambda path: path in wanted)]
        assert len(self.datasets[0]) == len(wanted), (len(self.datasets[0]), len(wanted))

        # CLIP's Food-101 class names are the folder names with spaces ("apple pie")
        self.num_classes = len(self.datasets[0].classes)
        self.classes = [c.replace('_', ' ') for c in self.datasets[0].classes]

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
