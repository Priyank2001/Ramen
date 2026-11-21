# Code design for experiments

How to add a method, variant or analysis without touching the repository's working code, organised so that a reader can find each piece of behaviour in one obvious place.

## Contents
1. What counts as shared (read-only) code
2. Experiment folder layout
3. Adding a method: local registry + harness subclass
4. When you must copy a file
5. SOLID, concretely
6. Checklist before a run

## 1. Read-only code

In the Ramen repo, treat these as read-only:
- `src/` (methods, datasets, models, `main.py`)
- `cfg/` (the YAML configs for existing methods)
- `new_experiments/ramen_analysis.py`, `ramen_examples.py`, `cache_viewer.py`, `confusion_shift.py` and the other shared helpers in `new_experiments/`
- earlier dated experiment folders (their notebooks are a record of what was run)

Reading, importing and subclassing these is fine. Editing them is not. Cached results in `new_experiments/results/` are produced by this code, so changing it makes those caches lie about what produced them.

## 2. Experiment folder layout

```
new_experiments/YYYY-MM-DD_<topic>/
  README.md          # question, table of every file and its job, how to run, results table,
                     # math of every method (baseline + each variant), why-it-does/doesn't-make-sense
  methods.py         # the new method classes (subclasses / mixins of repo methods)
  registry.py        # name -> class for the methods in this folder; falls back to the repo's registry
  experiment.py      # the experiment class: dataset/seed/variants, run or load, sanity check, tables
  configs.py         # VARIANTS = {name: (algo, cfg_overrides)}, DATASETS = {name: model/num_workers}
  analysis.py        # pure functions/classes over cached results (no GPU, no side effects)
  plots.py           # one function per figure
  run_<topic>.py     # CLI runner: dataset, seed, shard, n_shards, --smoke
  make_notebook.py   # writes the numbered notebooks from shared cells
  copied/            # only if needed: copies of repo files with a provenance header (section 4)
  logs/
  01_<topic>_<dataset>.ipynb
```

Split a file when it starts doing two jobs. Merge files that are only ever used together and are tiny. The README table must list every file.

## 3. Adding a method without editing the repo registry

The repo resolves methods by name via `methods.get_method_class`, and the harness calls it from `RamenExperiment.build_method(algo, cfg)`. Override that one method instead of editing either file.

```python
# registry.py
"""Methods defined in this experiment folder, looked up before the repo's own."""
from methods import get_method_class as repo_method_class  # src/ is on the path once ramen_analysis is imported

from local_methods import RamenFoo, RamenFooOracle  # this folder's methods.py, imported under a non-clashing name

LOCAL_METHODS = {cls.__name__: cls for cls in (RamenFoo, RamenFooOracle)}


def get_method_class(name):
    return LOCAL_METHODS.get(name) or repo_method_class(name)
```

```python
# experiment.py
from ramen_analysis import EXPERIMENTS
from registry import get_method_class


class LocalMethodsMixin:
    """Lets the harness build this folder's methods; every other behaviour is the harness's own."""

    def build_method(self, algo, cfg=None):
        # Same as RamenExperiment.build_method; only the class lookup differs
        args = SimpleNamespace(config={**self._load_cfg(algo), **(cfg or {})}, device=self.device,
                               max_batch_size=self.batch_size, use_tbn=False)
        return get_method_class(algo)(deepcopy(self.clip), self.datasets, args)


def make_local_experiment(dataset, **kwargs):
    base = EXPERIMENTS[dataset]
    cls = type(f'Local{base.__name__}', (LocalMethodsMixin, base), {})
    return cls(**kwargs)
```

Notes:
- The local package `methods` would shadow the repo's `src/methods`. Name the file something else (`local_methods.py`, `foo_methods.py`).
- The override repeats the four lines of `RamenExperiment.build_method` (`new_experiments/ramen_analysis.py`) instead of adding a hook to the harness. Keep it identical to the original apart from the lookup, so the "off" variant stays bitwise identical. If the harness version changes, check that the override still matches it.
- A local class whose name matches a repo method (e.g. `Ramen`) would shadow it. Give local methods distinct names.
- New configs: keep them as Python dicts in `configs.py`, passed as `cfg=` overrides on top of the repo's `cfg/<Dataset>/<BaseAlgo>.yaml`. Don't add YAML files under `cfg/`.
- Cache names must be unique to the experiment (prefix with the topic, e.g. `foo_t3_drop`) so they never collide with existing caches in `new_experiments/results/`.

## 4. When you must copy a file

Subclass when you can. Copy only when the change has to go in the middle of a long method that has no hook, or when the class hard-codes something that can't be overridden. Then:
1. Copy into `copied/<OriginalName>_<topic>.py`. Rename the class too (`RamenFooBase`) so the two can't be confused in a traceback.
2. Start the file with a header comment:
   ```python
   # Copied from src/methods/Ramen.py at commit 063343d (2026-11-01).
   # Changes: retrieval weights now take a per-entry prior (search "CHANGED").
   # Everything else is byte-identical to the source; diff against it to review.
   ```
3. Mark each changed block with `# CHANGED:` and a one-line reason.
4. Sanity check: with the change switched off, the copy must give 100.00% identical predictions to the original (section 2 of SKILL.md).

If the original file has a real bug, don't fix it in the repo. Tell the user what it is, what it affects (which cached results) and the proposed fix, and let them decide.

## 5. SOLID, concretely (in this codebase)

- **Single responsibility**: the method class adapts and predicts; logging internals goes in a recorder mixin; selection rules are their own small classes (`Selector.select(memory, query) -> ids`); analysis and plotting live in their own modules and never run the model.
- **Open/closed**: a new variant adds a class or a line in `VARIANTS`. If you're adding `if cfg['mode'] == 'x'` to an existing class, make a strategy class instead and pass it in.
- **Liskov**: a subclass takes the same constructor arguments and `forward` signature as its base, and its default config reproduces the base exactly. Anything that changes the contract (e.g. needs `labels=`) declares it (`records_labels = True`) the way the harness expects.
- **Interface segregation**: small mixins (`RecorderMixin`, `OracleLabelsMixin`), each with one feature, instead of one base class that does everything.
- **Dependency inversion**: classes receive the model, datasets, config and strategy objects. They don't import global state or construct their own data loaders. This keeps them testable on a 200-image subset.

Readability:
- Module docstring at the top of every file: what it is for, and a 3–6 line usage example.
- Class/method names say what they mean in the problem's terms.
- Comments explain *why*, at the density of the surrounding repo code. Don't add comments that repeat the code.
- Constants with meaning (capacity 75, β 5) come from config, not literals inside methods.
- Oracles (anything using true labels) say so in the class name or docstring and in the config comment.

## 6. Checklist before a run

- [ ] `git status` shows no modified files under `src/`, `cfg/` or the shared `new_experiments/*.py` from this experiment.
- [ ] New methods resolve through the local registry; `get_method_class('Ramen')` still returns the repo's class.
- [ ] Smoke run (`--smoke`, `n_per_domain=200`): "off" variant == Ramen at 100.00%.
- [ ] Cache names are prefixed with the experiment topic.
- [ ] README table lists every file and its job.
- [ ] README gives the math of each method and a results-based explanation of why it makes sense or not (SKILL.md section 5).
