#!/usr/bin/env bash
# Probability-saving re-runs for the stacking analysis: Ramen seeds 1-4 (seed 0's probs are already cached) and
# active_prob_T3 seeds 0-4, on DomainNet (~45 min each) and ImageNet-C 5K (~8 min each). One worker per GPU pulls the
# next job from a shared queue; ImageNet-C first so the sanity check (same predictions as the cached runs) reports early.
#   cd new_experiments/2026-10-25_stacking && setsid nohup bash run_queue.sh > logs/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=~/miniconda3/envs/RAMEN/bin/python
QUEUE=logs/queue.txt
: > "$QUEUE"
for ds in ImageNetC5K DomainNet; do
  for seed in 0 1 2 3 4; do echo "$ds $seed active_prob_T3" >> "$QUEUE"; done
  for seed in 1 2 3 4; do echo "$ds $seed Ramen" >> "$QUEUE"; done
done

worker() {
  local gpu=$1 job
  while true; do
    job=$(flock "$QUEUE.lock" bash -c "head -n1 '$QUEUE'; sed -i 1d '$QUEUE'")
    [ -z "$job" ] && break
    set -- $job
    echo "[$(date +%H:%M:%S)] gpu$gpu start $job"
    CUDA_VISIBLE_DEVICES=$gpu $PY run_probs.py "$1" "$2" "$3" > "logs/${1}_s${2}_${3}.log" 2>&1
    echo "[$(date +%H:%M:%S)] gpu$gpu done  $job (exit $?): $(grep 'same predictions' "logs/${1}_s${2}_${3}.log")"
  done
}
worker 0 & worker 1 & worker 2 &
wait
echo "[$(date +%H:%M:%S)] all jobs done"
