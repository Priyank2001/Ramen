# Code style: clean, readable, maintainable (OOP + SOLID)

Research code gets read far more often than it gets written: by the reviewer, by the advisor, and by you three weeks later when a number looks odd. These rules exist so that a reader can find where a behaviour lives, change it in one place, and trust that nothing else moved.

## The SOLID rules, applied to this repo

| principle | what it means here | example |
|---|---|---|
| **S**ingle responsibility | A class does one job; a module groups one concern. | `MemoryGuard` watches RAM and knows nothing about accuracy. `RecordWriter` writes records and knows nothing about Ramen. |
| **O**pen/closed | Add behaviour by adding a class and registering it, never by editing an `if/elif` chain or upstream code. | A new write policy is a new `WritePolicy` subclass plus `@register`; nothing else changes. |
| **L**iskov substitution | Any variant can be used where `Ramen` is expected: same constructor, same `forward(x) -> logits`, same `reset()`. | An oracle adds `uses_labels = True` and `forward(x, labels=None)`, and still works when called without labels if its strategy is label-free. |
| **I**nterface segregation | Small interfaces. Label access is a separate capability, not a parameter every method carries. | Only `LabelAware` methods receive labels; the runner checks the capability. |
| **D**ependency inversion | High-level code depends on `ramen_lab` interfaces. Only `ramen_lab/upstream.py` imports `src/`. | The runner takes a `MethodFactory`, a `StreamFactory` and a `MemoryGuard`, injected, not constructed inside it. |

## Extending a frozen upstream (src/ is read-only)

Prefer the first option that works:

1. **Configure:** pass a different config dict. Upstream reads `args.config`.
2. **Inject:** replace a collaborator through the constructor or a factory, e.g. a different cache class for `Ramen`'s queues, built by a `ramen_lab` subclass that overrides one construction step.
3. **Subclass and override one method:** a small override that calls `super()`.
4. **Wrap (decorator / adapter):** e.g. a `ByClassFeatureLogger` that wraps the model's `featurize` to record features.
5. **Re-implement one step:** only when the change sits inside upstream's monolithic `forward` (e.g. the write/retrieve order).
   - Re-implement it in `ramen_lab` as clearly named step methods (`compute_gradients`, `write_memory`, `retrieve`, `adapt_and_predict`). That is the Template Method pattern: variants override a step, not the loop.
   - It **must** have a test proving the off setting is bit-identical to upstream `Ramen` (`torch.equal` on logits over several batches with a filled cache), plus the same check on real data in the runner's `check` step.

Never copy an upstream file into `ramen_lab` "to tweak it". Never monkey-patch module globals in `src/` from experiment code. If a patch seems unavoidable, it belongs in `ramen_lab/upstream.py`, behind a named function with a test.

## Design patterns that fit TTA experiments

- **Strategy + registry** for interchangeable behaviour: write policy, retrieval rule, weighting, step rule. Name → class in one dict, filled by a `@register("name")` decorator; the config picks by name; an unknown name raises a `ValueError` that lists the options.
- **Template Method** for the adaptation loop: one `forward` that calls named steps.
- **Factory** for building runs from a `RunConfig`: model, datasets, method and stream are built in one place, so a run is reproducible from its config alone.
- **Frozen dataclasses** for configs and results (`@dataclass(frozen=True)`): runs can't mutate their config by accident, and `asdict()` goes straight into `metrics.json`.
- **Observer / hook** for logging internals: methods emit events (`on_write`, `on_retrieve`) to registered loggers, so logging never clutters the adaptation math.

## Readability rules

- **Names say what a thing is in the experiment's words:** `retrieved_grads`, `write_after_prediction`, `noise_floor`, not `tmp`, `g2`, `flag`. Symbols in the notebook derivation reuse these names (`G_i` ↔ `aggregated_grad`).
- **Type hints on every public function and method;** `Tensor` shapes go in the docstring: `(B, D) per-sample gradients`.
- **Docstrings say why and what, not how:** one line of purpose, then shapes and units, then anything non-obvious (fp16, self-matches, label use). Comments only where the code can't speak for itself.
- **Functions ≤ ~40 lines, classes ≤ ~200 lines.** If it's longer, a responsibility is hiding in it.
- **No magic numbers:** a constant with a name and a source comment (`QUEUE_CAPACITY = 300  # cfg/DomainNet/Ramen.yaml`).
- **No hidden global state:** no module-level mutable caches, no reliance on the global RNG. Seeded generators are passed explicitly.
- **Fail loudly, early:** validate configs in `__post_init__`; `raise` with a message saying what to do; no silent fallbacks that change results.
- **Pure where possible:** statistics and plotting functions take arrays or frames and return values or figures. No I/O inside, so the notebook and the tests can call them directly.
- **Imports:** standard library, third-party, `ramen_lab`, in that order. No `import *`. No `sys.path` hacks outside `ramen_lab/upstream.py` and the experiment's CLI entry point.

## Tests (pytest, in `tests/`)

- One test file per `ramen_lab` module. Name each test by the behaviour it checks (`test_reset_clears_memory`, not `test_1`).
- GPU tests use random images and 5 fake classes (no datasets), batch ≥ 32, and small caches. Mark them so CPU-only runs can skip them.
- Required for every method variant:
  - identity with upstream `Ramen` (off setting);
  - reset;
  - weights restored;
  - frozen parameters;
  - label independence (or a non-vacuous label test for an ORACLE);
  - determinism.
- The runner and recorder are tested on a 2-batch fake stream, including that the `MemoryGuard` stop is honoured.

## Skeleton (what "clean" looks like here)

```python
"""Write policies: which queue a sample enters and when. Ramen's own policy is ZeroShotBeforeRetrieval."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ramen_lab.registry import Registry

WRITE_POLICIES: Registry["WritePolicy"] = Registry("write policy")


@dataclass(frozen=True)
class WriteDecision:
    """keep (B,) bool: samples that enter memory; queue (B,) long: the class queue for each."""
    keep: Tensor
    queue: Tensor


class WritePolicy(ABC):
    """Decides which samples of a batch are written into memory, under which class, and when."""

    write_after_prediction: bool = False  # False: before retrieval (Ramen)
    uses_labels: bool = False             # True only for ORACLE policies

    @abstractmethod
    def decide(self, zs_logits: Tensor, adapted_logits: Optional[Tensor], labels: Optional[Tensor]) -> WriteDecision:
        """zs_logits (B, C); adapted_logits (B, C) or None if writing before prediction; labels (B,) only if uses_labels."""


@WRITE_POLICIES.register("ramen")
class ZeroShotBeforeRetrieval(WritePolicy):
    """Ramen: every sample, before retrieval, under its zero-shot class."""

    def decide(self, zs_logits, adapted_logits, labels):
        queue = zs_logits.argmax(dim=-1)
        return WriteDecision(keep=torch.ones_like(queue, dtype=torch.bool), queue=queue)
```
