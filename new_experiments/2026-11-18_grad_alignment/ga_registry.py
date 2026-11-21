"""Methods defined in this folder, looked up before the repo's own (src/methods/__init__.py is never edited)."""
from methods import get_method_class as repo_method_class  # src/ is on the path once ramen_analysis is imported

from ga_methods import RamenGradientProbe
from landscape_methods import RamenLandscapeProbe

LOCAL_METHODS = {'RamenGradientProbe': RamenGradientProbe, 'RamenLandscapeProbe': RamenLandscapeProbe}
BASE_CONFIG = {'RamenGradientProbe': 'Ramen', 'RamenLandscapeProbe': 'Ramen'}   # starts from cfg/<dataset>/Ramen.yaml; no YAML is added under cfg/


def get_method_class(name):
    return LOCAL_METHODS.get(name) or repo_method_class(name)


def config_name(name):
    return BASE_CONFIG.get(name, name)
