#!/usr/bin/env bash
# Waits for the 8 E3 result files (result files, not pgrep), then runs analysis_e3.py.
cd "$(dirname "$0")"
PY=/home/priyank1/miniconda3/envs/RAMEN/bin/python
R=../results
files() { for v in hide_top3_wrong_true hide_rand3_wrong_true hide_top10_wrong_true hide_rand10_wrong_true; do
  echo $R/DomainNet_clip_vitbase32_mixed_nall_bs100_s0_$v.pkl $R/ImageNetC5K_clip_vitbase16_mixed_nall_bs100_s0_$v.pkl; done; }
until [ "$(ls $(files) 2>/dev/null | wc -l)" -eq 8 ]; do sleep 60; done
sleep 60
$PY analysis_e3.py 2>&1 | grep -v -i warn | tee logs/e3_analysis.txt
