# 2026-10-25 — Stacking the confirmed fixes on ImageNet-C and DomainNet

Do the zero-shot-weighted active set (T3), output prior correction (pc) and causal kNN smoothing (knn) add up on the two large-label datasets? pc and knn are exact offline transforms of a run's saved probabilities. The only GPU work was re-running Ramen and T3 with the full softmax saved.

**Result.** The pre-registered full stack (λ = 2) fails: ImageNet-C −0.02 ± 0.11, DomainNet −1.13 ± 0.04 (seeds 0–4). kNN hurts on ImageNet-C, and kNN + pc at λ = 2 clash on both datasets. Post-hoc, T3 + pc with λ = 1 clears both noise floors on 5/5 seeds (+1.13 / +0.41). It needs a held-out dataset before it counts. Details in the notebook's interpretation section.

| File | What it is |
|---|---|
| `stacking.ipynb` | The write-up: question, data, sanity checks, results, additivity, per domain, post-hoc λ, mechanism, interpretation, glossary |
| `run_probs.py` | Re-runs Ramen / active_prob_T3 for one (dataset, seed) with `capture_probs=True`, saved as `<variant>_p`, and compares with the cached run |
| `run_queue.sh` | Job queue, one worker per GPU: Ramen seeds 1–4 + T3 seeds 0–4 on both datasets |
| `stack.py` | Offline stacking (`python stack.py <ds> 0 1 2 3 4`), summaries and plots; reproduces `quick_checks/knn_seeds.py` exactly on CIFAR-100-C |
| `build_notebook.py` | Writes and executes the notebook (`python build_notebook.py interpretation.md`) |
| `interpretation.md` | Text of the interpretation cell |
| `logs/` | Run and stacking logs |

Outputs: `results/<ds>_..._s<seed>_{Ramen,active_prob_T3}_p.pkl` + `_probs.npy` (all 18 re-runs give 100.00% the same predictions as the cached runs) and `results/stacking/<ds>_s<seed>_{stack,per_domain,knn_diag}.csv`.

Gotcha: `stack.py` on a GPU that is already running a Ramen job can OOM-kill that job (it did once; the job was re-queued). Run it with `CUDA_VISIBLE_DEVICES=""` while the GPUs are busy. That takes about 25 s per seed on ImageNet-C and about 10 min per seed on DomainNet.
