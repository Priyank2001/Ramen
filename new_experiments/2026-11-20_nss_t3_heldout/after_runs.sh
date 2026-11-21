#!/usr/bin/env bash
# Waits for all 10 held-out runs (result files, not pgrep), then writes tables/ and heldout.ipynb.
cd "$(dirname "$0")"
PY=/home/priyank1/miniconda3/envs/RAMEN/bin/python
R=../results/ImageNetVariants_clip_vitbase16_mixed_nall_bs100
until [ "$(ls ${R}_s{0,1,2,3,4}_nss_niqe_white_l0.1_probT3.pkl ${R}_s{0,1,2,3,4}_nss_niqe_white_l0.1.pkl 2>/dev/null | wc -l)" -eq 10 ]; do sleep 60; done
sleep 60  # let the last _stats / _probs finish writing
$PY analysis.py 2>&1 | grep -v -i warn | tee logs/analysis.txt
$PY build_notebook.py 2>&1 | tail -3 | tee logs/notebook.log
echo ALL DONE | tee -a logs/analysis.txt
