#!/usr/bin/env bash
# Held-out datasets for the frozen T3 + pc(lam=1) test: ImageNet-R, ImageNet-A, ImageNet-Sketch.
# Each archive goes to its own empty directory under ~/data/downloads/, is verified, checked for unsafe paths,
# then extracted into ~/data/imagenet_variants/. Resumable (curl -C -); re-running skips finished steps.
#   setsid nohup bash download.sh > logs/download.log 2>&1 < /dev/null &
set -u
D=~/data/downloads
OUT=~/data/imagenet_variants
mkdir -p "$OUT"

fetch() {  # name url expected_bytes
  local f="$D/$1/$(basename "$2")"
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$3" ]; then echo "[$(date +%H:%M:%S)] $1: already downloaded"; return 0; fi
  echo "[$(date +%H:%M:%S)] $1: downloading $2"
  for try in 1 2 3 4 5; do
    curl -L --fail --silent --show-error -C - --retry 5 --retry-delay 10 -o "$f" "$2" && break
    echo "[$(date +%H:%M:%S)] $1: curl failed (try $try), retrying"; sleep 30
  done
  local got; got=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$got" != "$3" ]; then echo "[$(date +%H:%M:%S)] $1: FAILED size $got != $3"; return 1; fi
  echo "[$(date +%H:%M:%S)] $1: size ok ($got bytes)"
}

safe_listing() {  # archive listing on stdin; fail on absolute paths or '..' components
  ! grep -qE '(^/|(^|/)\.\.(/|$))'
}

extract_tar() {  # name tarfile expected_top_dir expected_images
  local f="$D/$1/$2"
  if [ -d "$OUT/$3" ] && [ "$(find "$OUT/$3" -type f | wc -l)" = "$4" ]; then echo "[$(date +%H:%M:%S)] $1: already extracted"; return 0; fi
  tar -tf "$f" | safe_listing || { echo "[$(date +%H:%M:%S)] $1: UNSAFE paths in archive, not extracting"; return 1; }
  tar -xf "$f" -C "$OUT" --no-same-owner --no-same-permissions || { echo "[$(date +%H:%M:%S)] $1: tar FAILED"; return 1; }
  local n; n=$(find "$OUT/$3" -type f | wc -l)
  echo "[$(date +%H:%M:%S)] $1: extracted $n files, $(ls "$OUT/$3" | wc -l) class folders (expected $4 images)"
}

fetch imagenet-a https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar 687552512 && \
  extract_tar imagenet-a imagenet-a.tar imagenet-a 7501   # 7,500 images + README.txt

fetch imagenet-r https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar 2191079936 && \
  extract_tar imagenet-r imagenet-r.tar imagenet-r 30001  # 30,000 images + README.txt

SK=$D/imagenet-sketch/ImageNet-Sketch.zip
if fetch imagenet-sketch https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip 7593573012; then
  echo "[$(date +%H:%M:%S)] imagenet-sketch: checking sha256"
  if echo "4d3ce6b82a7c48aabf14aeac105eb533e75707f152cc858f55bfc022031a72ed  $SK" | sha256sum -c --quiet; then
    echo "[$(date +%H:%M:%S)] imagenet-sketch: sha256 ok"
    if [ -d "$OUT/sketch" ] && [ "$(find "$OUT/sketch" -type f | wc -l)" = "50889" ]; then
      echo "[$(date +%H:%M:%S)] imagenet-sketch: already extracted"
    elif unzip -Z1 "$SK" | safe_listing; then
      unzip -q -o "$SK" -d "$OUT" && \
        echo "[$(date +%H:%M:%S)] imagenet-sketch: extracted $(find "$OUT/sketch" -type f | wc -l) files (expected 50,889)"
    else
      echo "[$(date +%H:%M:%S)] imagenet-sketch: UNSAFE paths in archive, not extracting"
    fi
  else
    echo "[$(date +%H:%M:%S)] imagenet-sketch: sha256 MISMATCH"
  fi
fi
echo "[$(date +%H:%M:%S)] download.sh finished"
