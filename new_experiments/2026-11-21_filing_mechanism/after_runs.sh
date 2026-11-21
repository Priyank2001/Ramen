#!/usr/bin/env bash
# Waits for the 12 E2 result files (result files, not pgrep), then runs analysis_e2.py.
cd "$(dirname "$0")"
PY=/home/priyank1/miniconda3/envs/RAMEN/bin/python
R=../results
files() { for v in hide_samezs_wrong_true hide_random_wrong_true hide_samezs_all_true hide_samezs_all_zs; do
  echo $R/DomainNet_clip_vitbase32_mixed_nall_bs100_s0_$v.pkl $R/ImageNetC5K_clip_vitbase16_mixed_nall_bs100_s0_$v.pkl \
       $R/EuroSAT_clip_vitbase16_mixed_nall_bs100_s0_$v.pkl; done; }
until [ "$(ls $(files) 2>/dev/null | wc -l)" -eq 12 ]; do sleep 60; done
sleep 60
$PY analysis_e2.py 2>&1 | grep -v -i warn | tee logs/e2_analysis.txt
