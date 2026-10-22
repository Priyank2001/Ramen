# Handoff for PC2 (RTX 3080 Ti + RTX 2080 Ti): continue Ramen experiments

Background, in reading order:
1. [ramen-tta-thesis-brief.md](ramen-tta-thesis-brief.md): the thesis goal (beat Ramen on mixed-domain TTA), how Ramen works, the four improvement angles.
2. [ramen-experiments-handoff.md](ramen-experiments-handoff.md): earlier findings. **Its ablation section is stale.** Both ablations it lists as "not yet run" were run on 2026-09-25 on the main PC (3× RTX 3090), and the results change its conclusion. Use Section 1 below instead.

This file covers what PC2 should run, how to run it, and what to send back.

---

## 1. Current state (main PC, RTX 3090, seed 0, all from `log/`)

Mixed-domain baselines. CIFAR-10-C uses ViT-B/32 and CIFAR-100-C uses ViT-B/16, as in the paper.

| dataset | NoAdapt | Tent | Ramen |
|---|---|---|---|
| CIFAR-10-C | 61.36 | 66.71 | 72.63 |
| CIFAR-100-C | 37.10 | 41.14 | 46.12 |

**Ablation 1: `tta_mode` single vs. mixed** (single = memory reset after every domain). Mean of (single − mixed), in pp:

| | noise domains (3) | other 12 | average |
|---|---|---|---|
| CIFAR-10-C | −0.33 | +0.49 | +0.33 |
| CIFAR-100-C | −0.05 | −0.12 | −0.10 |

→ Single and mixed are about equal. Per-domain differences are within ±3 pp with no consistent sign. **Sharing one memory across domains in mixed mode does not measurably hurt Ramen.** The old handoff's reason for angle 4 ("noise-domain pseudo-labels pollute the shared memory") is not supported.

**Ablation 2: `beta` = 5 vs. 0** (distance weighting in α_ij):

| | Δavg (β5 − β0) | biggest gains |
|---|---|---|
| CIFAR-10-C | +2.57 | gaussian_noise +5.8, shot_noise +4.8, frost +4.5, fog +3.7 |
| CIFAR-100-C | +1.03 | all 15 domains improve (+0.2 … +1.9) |

→ Distance weighting helps the **noise domains most**. That contradicts the old handoff's "the embedding carries no class signal for noise domains, so retrieval can't help". Ramen already lifts gaussian_noise on CIFAR-10-C from 38.7 (NoAdapt) to 60.8. **Retrieval and weighting improvements (angles 1–3) are back on the table.** Angle 4 is still worth testing, but as "pseudo-label noise in general", not "cross-domain contamination".

**Open caveat:** there is no seed-to-seed variance yet, and every effect above is under 3 pp. Measuring that noise floor is PC2's first job.

---

## 2. PC2 setup

Hardware (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, the same order `nvidia-smi` shows):
- **GPU 0 = RTX 3080 Ti, 12 GB**
- **GPU 1 = RTX 2080 Ti, 11 GB**
- Driver 515.65.01, which supports CUDA ≤ 11.7.

This is the shared `narendra` account. **Do not uninstall anything from `~/.local`**, since it may belong to other users. It contains a CUDA-12 torch that breaks on this driver, so the env must ignore it.

```bash
conda activate RAMEN
conda env config vars set PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
conda deactivate && conda activate RAMEN
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
#   fallback if CUDA still fails to init:
#   pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
cd ~/priyank/Ramen && pip install -r requirements.txt
python -c "import torch; print(torch.__file__, torch.__version__, torch.version.cuda, torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```
The check must print a path under `anaconda3/envs/RAMEN`, `+cu118`, `True`, and both GPU names.

Files expected on PC2. If they're missing, copy them from the main PC with `rsync` (main PC paths: repo `~/Projects/Ramen`, data `~/data/corruption`, weights `~/.cache/clip`):
- repo → `~/priyank/Ramen`
- `~/data/corruption/CIFAR-10-C` (optionally `CIFAR-100-C`). `main.py`'s default `--data_root ~/data` resolves to `~/data/corruption/<name>`.
- `~/.cache/clip/` (CLIP weights; saves the download)

---

## 3. Rules for PC2 results

1. **Compare only within one GPU model.** The same seed gives slightly different numbers on the 3080 Ti, the 2080 Ti and the 3090, and the effects are 0.3–3 pp. Every job list below includes its own baseline on that GPU. Never compare a PC2 number with a 3090 number.
2. **One CSV per run.** `main.py` *appends* to `--save_to`. The queue script gives every run its own file under `log/pc2/<GPU>/<dataset>_<mode>/<tag>_seed<seed>.csv`.
3. **Batch size stays 100.** It controls how often the memory is updated, so changing it changes the method. If a run doesn't fit in memory, report it; don't shrink the batch.
4. **A difference counts only if it clearly exceeds the seed spread** of the baseline on the same GPU (roughly > 2× the max−min across seeds 0/1/2).

---

## 4. The queue runner and summarizer

Save as `shell/run_queue.sh`. It runs one job list sequentially on one GPU, writes a per-job config into `cfg_runs/<tag>/`, logs stdout and peak GPU memory, and skips jobs whose CSV already exists, so a crashed queue can simply be restarted. Tested on the main PC with stubbed `python`/`nvidia-smi` (config overrides, adding new keys, resume/skip).

```bash
#!/usr/bin/env bash
# Run a list of Ramen jobs sequentially on one GPU.
# Usage: bash shell/run_queue.sh <gpu_index> <jobs_file>
# jobs_file: one job per line, whitespace-separated, '#' comments allowed:
#   <tag> <dataset> <model> <tta_mode> <seed> <overrides>
#   overrides = comma-separated key=value applied to cfg/<dataset>/Ramen.yaml, or '-' for none
# Output: log/pc2/<gpu_name>/<dataset>_<mode>/<tag>_seed<seed>.{csv,log,mem}
# A job whose .csv already exists is skipped, so the queue can be re-run after a crash.
set -uo pipefail
GPU=$1; JOBS=$(realpath "$2")
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1
GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | sed 's/NVIDIA //; s/GeForce //; s/ /_/g')
cd "$REPO/src"

grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | while read -r tag dataset model mode seed overrides; do
  out="$REPO/log/pc2/$GPU_NAME/${dataset}_${mode}"
  base="$out/${tag}_seed${seed}"
  [ -f "$base.csv" ] && { echo "skip $base.csv (exists)"; continue; }
  mkdir -p "$out"

  cfgdir="$REPO/cfg_runs/$tag"
  mkdir -p "$cfgdir/$dataset"
  cfg="$cfgdir/$dataset/Ramen.yaml"
  cp "$REPO/cfg/$dataset/Ramen.yaml" "$cfg"
  if [ "$overrides" != "-" ]; then
    IFS=',' read -ra kvs <<< "$overrides"
    for kv in "${kvs[@]}"; do
      k=${kv%%=*}; v=${kv#*=}
      if grep -q "^$k:" "$cfg"; then sed -i "s/^$k:.*/$k: $v/" "$cfg"; else echo "$k: $v" >> "$cfg"; fi
    done
  fi

  echo "[$(date '+%F %T')] GPU$GPU ($GPU_NAME) start $tag $dataset $mode seed=$seed overrides=$overrides"
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$base.mem" &
  mon=$!
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU python main.py --dataset "$dataset" --model "$model" --tta_algo Ramen \
    --tta_mode "$mode" --seed "$seed" --cuda --batch_size 100 --config "$cfgdir" \
    --save_to "$base.csv" > "$base.log" 2>&1 < /dev/null
  rc=$?
  kill $mon 2>/dev/null
  peak=$(sort -n "$base.mem" | tail -1)
  echo "[$(date '+%F %T')] done rc=$rc  $(( ($(date +%s)-start)/60 )) min  peak_mem=${peak}MiB  $(grep '^total:' "$base.log")"
  [ $rc -ne 0 ] && { rm -f "$base.csv"; echo "  FAILED, see $base.log"; }
done
```

Save as `src/summarize.py`. It reads every result CSV under the given directories and prints average / noise-domain mean / other-domain mean per run, plus a full per-domain CSV. Tested on the main PC's `log/ablations/`.

```python
"""Summarize Ramen result CSVs: python summarize.py [log_dir ...]  (default: ../log/pc2)"""
import csv, glob, os, sys
import pandas as pd

NOISE = ['gaussian_noise', 'shot_noise', 'impulse_noise']

rows = []
for root in sys.argv[1:] or ['../log/pc2']:
    for path in sorted(glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)):
        with open(path) as f:
            lines = [r for r in csv.reader(f) if r]
        # each run appends 4 rows: args, config, domain header, accuracies -> keep the last run
        args, cfg, header, accs = lines[-4:]
        info = dict(kv.split('=', 1) for kv in args + cfg if '=' in kv)
        acc = dict(zip(header, map(float, accs)))
        other = [d for d in header if d not in NOISE + ['Average']]
        rows.append({
            'file': os.path.relpath(path, root),
            'dataset': info.get('dataset'), 'mode': info.get('tta_mode'), 'seed': info.get('seed'),
            'beta': info.get('beta'), 'topk': info.get('topk'), 'cap': info.get('max_capacity'),
            'avg': acc['Average'],
            'noise': round(sum(acc[d] for d in NOISE) / 3, 2),
            'other': round(sum(acc[d] for d in other) / len(other), 2),
            **{d: acc[d] for d in header if d != 'Average'},
        })

df = pd.DataFrame(rows)
pd.set_option('display.width', 250, 'display.max_columns', 30)
print(df[['file', 'dataset', 'mode', 'seed', 'beta', 'topk', 'cap', 'avg', 'noise', 'other']].to_string(index=False))
df.to_csv('summary_all_domains.csv', index=False)
print('\nper-domain table written to summary_all_domains.csv')
```

---

## 5. What to run on PC2 (Phase 0 noise floor + Phase 2 sweeps, CIFAR-10-C)

All of PC2's accuracy work is **CIFAR-10-C, ViT-B/32, mixed mode**. The Ramen memory is preallocated at 10 classes × 7500 × 39,936 grad dims × fp16 ≈ **6.0 GB**, which fits both cards. CIFAR-100-C / ViT-B/16 stays on the 3090s, except for one memory probe at the end of the 3080 Ti queue.

**Step 0 — smoke test (one job, ~minutes).** Run the first line of the GPU-1 list alone and check that `.log` ends with 15 per-domain accuracies plus `total:`, and that `peak_mem` is well under 11 GB. Then start the queues.

`shell/jobs_gpu1_2080ti.txt`: seed noise floor + β sweep
```
# tag     dataset   model          mode  seed overrides
base      CIFAR10C  clip_vitbase32 mixed 0    -
base      CIFAR10C  clip_vitbase32 mixed 1    -
base      CIFAR10C  clip_vitbase32 mixed 2    -
beta0     CIFAR10C  clip_vitbase32 mixed 0    beta=0.0
beta2.5   CIFAR10C  clip_vitbase32 mixed 0    beta=2.5
beta10    CIFAR10C  clip_vitbase32 mixed 0    beta=10.0
beta20    CIFAR10C  clip_vitbase32 mixed 0    beta=20.0
```

`shell/jobs_gpu0_3080ti.txt`: its own baseline + topk / memory-size sweep + CIFAR-100-C memory probe
```
# tag     dataset    model          mode  seed overrides
base      CIFAR10C   clip_vitbase32 mixed 0    -
base      CIFAR10C   clip_vitbase32 mixed 1    -
topk10    CIFAR10C   clip_vitbase32 mixed 0    topk=10
topk25    CIFAR10C   clip_vitbase32 mixed 0    topk=25
topk100   CIFAR10C   clip_vitbase32 mixed 0    topk=100
cap2500   CIFAR10C   clip_vitbase32 mixed 0    max_capacity=2500
cap5000   CIFAR10C   clip_vitbase32 mixed 0    max_capacity=5000
# last: does CIFAR-100-C / ViT-B/16 at batch 100 fit in 12 GB? (result comparable only to other 3080 Ti runs)
base      CIFAR100C  clip_vitbase16 mixed 0    -
```
Larger memory (e.g. `max_capacity=15000`) is left out on purpose: that preallocates about 12 GB and won't fit on these cards.

Launch both in tmux:
```bash
cd ~/priyank/Ramen && conda activate RAMEN
tmux new -d -s q1 "bash shell/run_queue.sh 1 shell/jobs_gpu1_2080ti.txt 2>&1 | tee log/pc2_queue_gpu1.txt"
tmux new -d -s q0 "bash shell/run_queue.sh 0 shell/jobs_gpu0_3080ti.txt 2>&1 | tee log/pc2_queue_gpu0.txt"
tail -f log/pc2_queue_gpu*.txt
```
Run time per CIFAR-10-C job is unknown on these cards. It was about 7–12 minutes on a busy 3090, so expect both queues to finish overnight.

Then summarize:
```bash
cd ~/priyank/Ramen/src && python summarize.py ../log/pc2
```

### How to read the results
- **Noise floor:** max−min of `avg`, `noise`, `other` across `base` seeds on the 2080 Ti (and seeds 0/1 on the 3080 Ti). Anything below about 2× that range is noise.
- **β sweep:** if accuracy keeps rising past β=5 (especially the `noise` column), the fixed β is under-tuned. Because distances differ between domains, that argues for a per-query adaptive β (candidate C below).
  If β=5 is the peak, Ramen's weighting is already well tuned, and adaptive weighting is less promising.
- **topk / memory size:** the paper claims robustness. Clear gains (> noise floor) from a non-default value support angle 3 (adaptive k / K). Flat curves mean angle 3 isn't worth the effort.

---

## 6. Next, once the queues are running (code work)

Make code changes on a git branch (`git checkout -b pc2-<topic>`), never directly on `experiments`, so they can be merged back into the main PC's copy cleanly. Every change must be **off by default via a config key**, so the unmodified config still reproduces the baseline exactly.

### Phase 1 — memory diagnostics (logging only, no effect on predictions)
- In `mixed_domain_test` ([main.py](main.py)), hand the true `label` and `domain_idx` to the TTA model, e.g. by setting attributes before `tta_model(image)`. Do this behind a `--diag` flag.
- Inside `Ramen.forward` ([methods/Ramen.py](methods/Ramen.py)), log per batch, split into noise vs. other domains:
  1. memory purity: fraction of stored entries whose pseudo-label is correct (needs the true label and domain stored alongside each entry);
  2. same-domain rate of the retrieved top-k neighbours;
  3. pseudo-label accuracy of the retrieved support set;
  4. how much of the aggregated gradient comes from the true class's queue vs. the other queues.
- This decides what comes next. Low purity → work on the memory (candidates A/B). Pure memory but low same-domain rate → work on retrieval (C/D).

### Candidate A — write memory entries with the *adapted* label (first code candidate)
Currently each sample is filed under `init_preds` (zero-shot, about 61% accurate on CIFAR-10-C) *before* adaptation ([Ramen.py:128-133](methods/Ramen.py#L128-L133)). The adapted prediction is about 73% accurate.

Idea: retrieve → adapt → predict → *then* write the sample into the queue of its adapted prediction. The stored gradient is still valid, because the model is reset to base weights for every batch, so `g_j` is always computed at the base weights.

Pitfalls:
- **Empty memory on the first batch:** `num_class_sum` would be 0, causing a divide-by-zero.
- **Self-inclusion:** in the original, each sample retrieves itself (distance 0) and its batch-mates, because the whole batch is written before retrieval. Delayed writing removes that. Add the sample's own gradient back explicitly, or accept the change and measure it separately.
- **Keep the ablation clean.** Run A0 = delayed write with `init_preds` (only the ordering changes) and A1 = delayed write with adapted labels, so the label effect is separated from the ordering effect.

Config key, e.g. `write_label: init | adapted` and `delayed_write: false | true`.

### Candidate B — confidence-aware memory
- Evict by `recency − λ·entropy` instead of pure arrival order. Currently `priorities` is just an arrival counter ([Ramen.py:124](methods/Ramen.py#L124)).
- Or only admit samples with `H < τ_H · ln C`.
- Separately, sharpen the confidence weight `exp(-H)` to `exp(-H/τ)`. With C=10 it only spans 1 → 0.1, which barely separates confident from unconfident samples.

### Candidate C — self-tuning distance weighting (angle 3)
Replace the fixed β with a per-query scale, e.g. `exp(-d / median(d_topk))`, since isolated noise-domain clusters have a different distance scale. Worth doing only if the β sweep in Section 5 shows β is under-tuned or domain-dependent.

Evaluate each candidate on CIFAR-10-C mixed, seeds 0/1/2, on the **same GPU** as its baseline, and report `avg` / `noise` / `other`.

---

## 7. Send back to the main PC

- `log/pc2/` (all CSVs, `.log`, `.mem`) and the `summarize.py` output. From the main PC:
  `rsync -av narendra@<pc2-host>:~/priyank/Ramen/log/pc2/ ~/Projects/Ramen/log/pc2/`
- Peak memory of the CIFAR-100-C probe on the 3080 Ti, which decides whether CIFAR-100-C work can move to PC2.
- Any code branch (`git bundle create pc2.bundle pc2-<topic>` is easiest if there's no shared remote).
- Don't cite anything as a result unless its CSV exists under `log/pc2/`.
