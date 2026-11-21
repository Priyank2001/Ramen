from .corruption import CORRUPTION_DATASETS, get_corruption_dataset_class
from .domainbed import DOMAINBED_DATASETS, get_domainbed_dataset_class
from .eurosat import EuroSAT
from .imagenet_variants import ImageNetVariants


def get_dataset_class(dataset_name):

    if dataset_name in CORRUPTION_DATASETS:
        return get_corruption_dataset_class(dataset_name)

    elif dataset_name in DOMAINBED_DATASETS:
        return get_domainbed_dataset_class(dataset_name)

    elif dataset_name == 'EuroSAT':
        return EuroSAT

    elif dataset_name == 'ImageNetVariants':
        return ImageNetVariants

    else:
        raise Exception("Unknown dataset name")
