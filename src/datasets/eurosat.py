import os
from torchvision.datasets import ImageFolder

# Readable class names for the CLIP prompts (CoOp's names, used by the CLIP TTA literature), keyed by folder name
EUROSAT_CLASSNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake',
}


class EuroSAT:
    """
    EuroSAT RGB (Sentinel-2 land cover): 27,000 64x64 images, 10 classes, a single domain.
    Expects root/EuroSAT/2750/<class folder>/*.jpg (the official EuroSAT.zip, md5 c8fa014336c82ac7804f0398fcb19387).
    """

    def __init__(self, root, transform=None):
        path = os.path.join(root, "EuroSAT", "2750")
        self.environments = ['satellite']
        self.datasets = [ImageFolder(path, transform=transform)]

        self.num_classes = len(self.datasets[-1].classes)
        self.classes = [EUROSAT_CLASSNAMES[cls] for cls in self.datasets[-1].classes]

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
