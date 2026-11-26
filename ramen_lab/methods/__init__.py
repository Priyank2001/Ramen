"""
Method registry: upstream NoAdapt / Tent / Ramen, plus our variants (added by importing their modules below).

A method follows upstream's interface: `Method(model, datasets, args)`, `forward(x) -> logits`, `reset()`.
A method that reads ground truth (an ORACLE) sets the class attribute `uses_labels = True` and takes
`forward(x, labels=None)`. The runner passes labels to no other method.
"""
from ramen_lab import upstream
from ramen_lab.registry import Registry

METHODS: Registry[type] = Registry("method")

for _name in upstream.UPSTREAM_METHODS:
    METHODS.add(_name, upstream.method_class(_name))


def uses_labels(method: object) -> bool:
    """True only for ORACLE methods, which declare it explicitly."""
    return bool(getattr(method, "uses_labels", False))


from ramen_lab.methods import grad_alignment  # noqa: E402,F401  (registers RamenGradAlignment)
