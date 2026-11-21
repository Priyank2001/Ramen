#!/usr/bin/env bash
# Seeds 1-4 for the hide-confident-wrong oracle and its random control (the variants that cleared the seed-0 screen
# on DomainNet; ImageNet-C added because it is cheap). One worker per GPU pulls the next job from a shared queue,
# longest jobs (DomainNet, ~47 min) first, ImageNet-C (~12 min) after.
#   cd new_experiments/2026-10-23_cache_contamination && setsid nohup bash run_seed_sweep.sh > logs/seed_sweep.log 2>&1 < /dev/null &
cd "$(dirname "$0")"
PY=~/miniconda3/envs/RAMEN/bin/python
QUEUE=logs/seed_sweep.queue
: > "$QUEUE"
for ds in DomainNet ImageNetC5K; do
  for seed in 1 2 3 4; do
    for v in oracle_drop_cw oracle_random_like_cw; do echo "$ds $seed $v" >> "$QUEUE"; done
  done
done

worker() {
  local gpu=$1 job
  while true; do
    job=$(flock "$QUEUE.lock" bash -c "head -n1 '$QUEUE'; sed -i 1d '$QUEUE'")
    [ -z "$job" ] && break
    set -- $job
    echo "[$(date +%H:%M:%S)] gpu$gpu start $job"
    CUDA_VISIBLE_DEVICES=$gpu $PY run_oracles.py "$1" "$2" 0 1 --variants "$3" > "logs/${1}_s${2}_${3}_gpu${gpu}.log" 2>&1
    echo "[$(date +%H:%M:%S)] gpu$gpu done  $job (exit $?)"
  done
}
worker 0 & worker 1 & worker 2 &
wait
echo "[$(date +%H:%M:%S)] all jobs done"
