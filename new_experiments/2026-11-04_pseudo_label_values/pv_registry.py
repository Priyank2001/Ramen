"""Methods of this folder and of 2026-11-01_label_oracle, looked up before the repo's own (src/ is never edited)."""
from registry import config_name as label_config_name  # 2026-11-01_label_oracle/registry.py (read-only import)
from registry import get_method_class as label_method_class

from pseudo_methods import RamenPseudoValues

LOCAL_METHODS = {'RamenPseudoValues': RamenPseudoValues}
BASE_CONFIG = {'RamenPseudoValues': 'Ramen'}  # starts from cfg/<dataset>/Ramen.yaml; no YAML is added under cfg/


def get_method_class(name):
    return LOCAL_METHODS.get(name) or label_method_class(name)


def config_name(name):
    return BASE_CONFIG.get(name) or label_config_name(name)
