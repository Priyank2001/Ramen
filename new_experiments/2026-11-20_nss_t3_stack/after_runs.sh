#!/usr/bin/env bash
# Waits for the 5 stack runs (result files, not pgrep), then writes tables/ and stack.ipynb.
cd "$(dirname "$0")"
PY=/home/priyank1/miniconda3/envs/RAMEN/bin/python
R=../results/ImageNetC5K_clip_vitbase16_mixed_nall_bs100
until [ "$(ls ${R}_s{0,1,2,3,4}_nss_niqe_white_l0.1_probT3.pkl 2>/dev/null | wc -l)" -eq 5 ]; do sleep 60; done
sleep 30  # let the last _stats.pkl finish writing
$PY analysis.py 2>&1 | grep -v -i warn | tee logs/analysis.txt
$PY build_notebook.py 2>&1 | tail -3 | tee logs/notebook.log
echo ALL DONE | tee -a logs/analysis.txt
