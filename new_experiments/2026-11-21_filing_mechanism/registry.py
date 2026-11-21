"""Methods defined in this experiment folder, looked up before the repo's own (src/methods is never edited)."""
from methods import get_method_class as repo_method_class  # src/methods, on the path once ramen_analysis is imported

from local_methods import RamenHideConfusion

LOCAL_METHODS = {'RamenHideConfusion': RamenHideConfusion}
# local method -> the repo method whose cfg/<dataset>/<name>.yaml it starts from (no YAML is added under cfg/)
BASE_CONFIG = {'RamenHideConfusion': 'Ramen'}


def get_method_class(name):
    return LOCAL_METHODS.get(name) or repo_method_class(name)


def config_name(name):
    return BASE_CONFIG.get(name, name)
