# Cache compression: storing Ramen's gradients in 8, 4 or 1 bit (2026-11-18)

**Question.** Ramen's per-class cache stores, for each past image, a 512-d CLIP key and a 39,936-d fp16 entropy
gradient over the LayerNorm parameters. The gradients are 99% of the cache (5.65 GB on ImageNet-C, 7.80 GB on
DomainNet). Can they be stored in fewer bits, and is that a better use of memory than keeping fewer images per class?

**Pre-registered rule** (in `cc_configs.py`, before any run). A codec is *lossless-enough* if |acc − Ramen| ≤ noise
floor, and *beats shrinking* if acc − acc(fp16 Ramen with a queue cut to the same bytes) > floor. Worth using only if
both hold. Floors: DomainNet 0.11, ImageNet-C 0.54, EuroSAT 1.51 (domain-averaged points).

## Result (seed 0, full streams)

Accuracy minus Ramen, domain-averaged points. "Queue" is fp16 Ramen with a shorter queue at the same memory.

| Memory | ImageNet-C codec / queue | DomainNet codec / queue | EuroSAT codec / queue |
|---|---|---|---|
| int8, 53.7% | +0.04 / −0.08 | +0.00 / −0.08 | −0.02 / +0.13 |
| int4, 29.0% | +0.03 / −0.25 | +0.01 / **−0.22** | −0.04 / +0.42 |
| sign, 10.5% | −0.10 / **−0.74** | −0.03 / **−0.52** | +0.17 / **+1.64** |

Bold = outside the floor. In images: DomainNet sign −128 of 596,006 vs its queue −3,433; ImageNet-C sign −71 of 75,000
vs −551; EuroSAT queue-789 +444 of 27,000.

- **Lossless-enough: 9 of 9** compressed runs. Each predicts the same class as Ramen on 91–99.7% of images.
- **Beats shrinking: 3 of 9**: 1-bit on ImageNet-C (+0.64) and DomainNet (+0.49), 4-bit on DomainNet (+0.22).
  int8 never does: half the queue is also within the floor.
- **EuroSAT goes the other way.** A shorter queue *helps* there (+1.64 at K = 789, just above the 1.51 floor). The
  K = 7,500 in its config was copied from CIFAR-10-C and never tuned. This is post-hoc, from one seed.
- **Memory.** The 1-bit cache is 9.5× smaller. Peak GPU memory drops from 13.5 to 8.4 GB on ImageNet-C and EuroSAT,
  and from 10.2 to 3.3 GB on DomainNet.
- **Speed cost (unplanned).** Decoding on every retrieval in plain PyTorch made DomainNet runs 1.8–2.4× slower
  (69–94 vs 39 min) and ImageNet-C runs 1.4–1.5× slower. This was not optimised.

**Why it works.** Ramen steps along sign(Σ w_j g_j). Quantisation error is small and unrelated across the retrieved
entries, so it mostly cancels in the sum. It can only flip coordinates whose sum is already near zero. 1-bit storage
keeps each sign and each 64-value block's size, which is close to what a sign-of-sum step needs.

## Correctness evidence

- `cc_off` (fp16 codec through the whole new code path) equals Ramen on **75,000 / 75,000**, **596,006 / 596,006** and
  **27,000 / 27,000** images. The 200-per-domain smoke runs were also identical.
- `verify.py` runs **86 real-data checks, all passing** (`logs/verify.log`):
  - every run saw the same stream;
  - logged cache bytes equal classes × K × bytes-per-entry, derived from the format by hand;
  - matched controls are within one slot of their codec;
  - one codec update per batch;
  - table accuracies equal a plain recount;
  - no file under `src/` or `cfg/` changed.
- `test_compression.py` runs **16 unit tests on the GPU, all passing** (`logs/unittest.log`):
  - error bounds for int8/int4;
  - exact 4-bit and 1-bit packing;
  - sign decodes to sign × block mean;
  - the fp16 cache equals Ramen's `PriorityCache` through evictions;
  - the int8 cache retrieves the same entries and returns exactly decode(encode(value));
  - byte accounting.

## Files

| File | Role |
|---|---|
| `grad_codecs.py` | `IdentityCodec`, `Int8Codec`, `Int4Codec`, `SignCodec`: block-wise allocate / write / read / bytes |
| `compressed_cache.py` | `CompressedPriorityCache(PriorityCache)`: values go through a codec, add/query inherited unchanged |
| `cc_methods.py` | `RamenCompressed(Ramen)`: swaps in the compressed caches; logs cache bytes, zero fraction, peak memory |
| `cc_configs.py` | datasets, floors, capacities, memory ratios, matched capacities, the 7 variants, the pre-registered rule |
| `cc_registry.py` | local method lookup (no edit to `src/methods/__init__.py`) |
| `cc_experiment.py` | `CompressionExperiment`: run / load, `sanity_check`, `table` |
| `run_compression.py` | CLI: `--smoke smoke` (200 per domain) or `--runs a,b,c` |
| `test_compression.py`, `verify.py` | unit tests (synthetic) and real-data checks |
| `plots.py`, `interpretation.py`, `make_notebook.py` | figures, shared interpretation text, notebook generator |
| `01_cache_compression.ipynb` | executed report: data, correctness, results, verdicts, mechanism, costs, glossary |
| `results_<dataset>_s0.csv` | final tables |
| `logs/` | run logs (timings), `verify.log`, `unittest.log`, smoke logs |
| `smoke/` | smoke-run results |

Full results are in `new_experiments/results/<prefix>_cc_*.pkl` and `_stats.pkl`. To reproduce:

```
CUDA_VISIBLE_DEVICES=0 python run_compression.py ImageNetC5K 0 --smoke smoke
CUDA_VISIBLE_DEVICES=0 setsid nohup python run_compression.py ImageNetC5K 0 --runs cc_off,cc_int8,... > logs/x.log 2>&1 &
python verify.py | tee logs/verify.log; python -m unittest -v test_compression
python make_notebook.py; jupyter nbconvert --to notebook --execute --inplace 01_cache_compression.ipynb
```

## Next

1. Seeds 1–4 for `cc_sign` and `cc_cap_sign` on ImageNet-C and DomainNet (the passes with the largest margins).
2. A EuroSAT capacity sweep with seeds (K = 7,500 / 2,000 / 800 / 200). This is the one lead here that could change
   accuracy rather than memory.
3. A fused decode, if speed matters.
