#!/usr/bin/env bash
# Held-out runs: NoAdapt (seed 0; it is stateless), Ramen and active_prob_T3 seeds 0-4 on ImageNetVariants.
# One worker per GPU pulls the next job from a shared queue.
#   cd new_experiments/2026-10-25_heldout && setsid nohup bash run_queue.sh > logs/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=~/miniconda3/envs/RAMEN/bin/python
QUEUE=logs/queue.txt
: > "$QUEUE"
echo "0 NoAdapt" >> "$QUEUE"
for seed in 0 1 2 3 4; do
  for run in Ramen active_prob_T3; do echo "$seed $run" >> "$QUEUE"; done
done

worker() {
  local gpu=$1 job status
  while true; do
    job=$(flock "$QUEUE.lock" bash -c "head -n1 '$QUEUE'; sed -i 1d '$QUEUE'")
    [ -z "$job" ] && break
    set -- $job
    echo "[$(date +%H:%M:%S)] gpu$gpu start $job"
    CUDA_VISIBLE_DEVICES=$gpu $PY run_heldout.py "$1" "$2" > "logs/s${1}_${2}.log" 2>&1
    status=$?
    echo "[$(date +%H:%M:%S)] gpu$gpu done  $job (exit $status): $(grep 'domain-avg' "logs/s${1}_${2}.log")"
  done
}
worker 0 & worker 1 & worker 2 &
wait
echo "[$(date +%H:%M:%S)] all jobs done"
