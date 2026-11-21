#!/usr/bin/env bash
# Sequential-stream runs, longest first (DomainNet), one worker per GPU pulling from a shared queue.
#   cd new_experiments/2026-10-26_sequential && setsid nohup bash run_queue.sh > logs/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=~/miniconda3/envs/RAMEN/bin/python
QUEUE=logs/queue.txt
: > "$QUEUE"
echo "ImageNetC5K 0 Ramen_cdistcheck" >> "$QUEUE"   # patched cdist (fp16 mm path) must reproduce the cached mixed Ramen
# (the sequential NoAdapt sanity run, ImageNetC5K seed 0, already passed: 75,000 of 75,000 identical)
for ds in DomainNet ImageNetVariants ImageNetC5K; do
  for run in Ramen active_prob_T3; do
    for seed in 0 1 2 3 4; do echo "$ds $seed $run" >> "$QUEUE"; done
  done
done
for ds in DomainNet ImageNetVariants ImageNetC5K; do
  for seed in 0 1 2 3 4; do echo "$ds $seed Ramen_reset" >> "$QUEUE"; done
done

worker() {
  local gpu=$1 job status
  while true; do
    job=$(flock "$QUEUE.lock" bash -c "head -n1 '$QUEUE'; sed -i 1d '$QUEUE'")
    [ -z "$job" ] && break
    set -- $job
    echo "[$(date +%H:%M:%S)] gpu$gpu start $job"
    CUDA_VISIBLE_DEVICES=$gpu $PY run_sequential.py "$1" "$2" "$3" > "logs/${1}_s${2}_${3}.log" 2>&1
    status=$?
    echo "[$(date +%H:%M:%S)] gpu$gpu done  $job (exit $status): $(grep 'domain-avg' "logs/${1}_s${2}_${3}.log")"
  done
}
worker 0 & worker 1 & worker 2 &
wait
echo "[$(date +%H:%M:%S)] all jobs done"
