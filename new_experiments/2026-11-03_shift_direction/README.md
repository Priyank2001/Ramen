# Shift direction (2026-11-03)

When Ramen's one-step adaptation leaves a prediction wrong, what kind of embedding move was it: a clean,
targeted move toward the final wrong class and away from the true/zero-shot one, or something larger and mostly
unrelated to that specific class pair? And when many images end up at the same wrong class, do they move in a
shared direction (a real attractor) or each their own way?

Pure analysis over data already recorded by `2026-10-28_confusion_records` (`ImageRecords`: `emb_before`,
`emb_after`, outcome labels) and the CLIP text features cached by `2026-11-01_pillow_sink` (`TextFeatures`). No
GPU, no new runs, nothing here changes Ramen.

| file | responsibility |
|---|---|
| `analysis.py` | `ShiftDecomposition`: per-image shift decomposed along each image's own (from, to) class-text axis (`explained_frac`, `delta_margin`), and destination coherence (`R`) with a bootstrap null |
| `plots.py` | explained-fraction / delta-margin box plots by outcome, coherence dot plot with null band |
| `make_notebook.py` | writes `01_shift_direction_domainnet.ipynb` |

Run (no GPU needed; both runs are already cached for DomainNet, seed 0):
```
cd new_experiments/2026-11-03_shift_direction
python make_notebook.py
~/miniconda3/envs/RAMEN/bin/jupyter nbconvert --to notebook --execute --inplace 01_shift_direction_domainnet.ipynb
```

## Headline (seed 0, full DomainNet, 596,006 images; 291,468 with a non-trivial zero-shot → Ramen outcome)

- **The shift is large but almost never targeted at the specific class pair that changed.** Median
  `explained_frac` (share of the shift's squared length on the image's own from→to text axis) is 0.003–0.016
  across outcomes — essentially none of the move is "about" the pair that flipped.
- **Yet the small projection onto that axis is almost always in the helpful direction.** Median `Δmargin` is
  positive for every outcome (+1.06 to +4.21), and positive for 73–100% of images per group. The flip rides a
  small, consistently-signed sliver of a much bigger move.
- **Images landing at the same wrong class move coherently (194 of 260 classes with ≥300 wrong images have R
  above their own bootstrap null; R up to 0.74), but not because of that class's own text-embedding pull**
  (`semantic_cos` only 0.06–0.21). **It is a domain effect**: destinations with the highest coherence (pillow,
  mushroom, helmet, swan, pear) are 89–95% quickdraw-domain images, and quickdraw's wrong images as a whole have
  R = 0.627 vs 0.39–0.46 for the other five domains — far above a bootstrap null built from the (quickdraw-heavy)
  pool of all wrongly-predicted images.
