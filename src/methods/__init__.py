from .NoAdapt import NoAdapt
from .Tent import Tent
from .Ramen import Ramen
from .RamenAnchor import RamenAnchor
from .RamenLog import RamenLog
from .RamenFix import RamenClean, RamenNSGD
from .RamenCache import RamenCache
from .RamenActive import RamenActive
from .RamenPolicy import RamenPolicy
from .RamenQuality import RamenQuality
from .RamenNSS import RamenNSS


def get_method_class(method_name):
    if method_name not in globals():
        raise NotImplementedError("Method not found: {}".format(method_name))
    return globals()[method_name]
