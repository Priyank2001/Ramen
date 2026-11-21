**H1 fails on DomainNet.** T3's gain over Ramen survives the sequential stream on two of three datasets:
- ImageNet-C **+0.52 ± 0.06** (floor 0.37, 5/5 above; mixed +0.62).
- ImageNet-R/A/Sketch **+0.66 ± 0.11** (floor 0.40, 5/5; mixed +0.59).
- DomainNet **−0.07 ± 0.02** (floor 0.13, 0/5 above, every per-seed CI below or across zero; mixed +0.24).
- With fp32 priorities (the amendment) DomainNet is also negative, **−0.12 ± 0.01** (0/5). So the DomainNet failure is not caused by the fp16 freeze. Under the pre-registered rule, T3 is **not** confirmed as a general improvement for domain-sequential streams.

**Where T3 loses on DomainNet.** With fp32, T3 − Ramen is negative in all six domains (−0.06 to −0.17), including clipart, the first domain, where the memory holds only clipart. In the mixed stream it was positive in five of six (+0.16 to +0.52). One reading, which is untested: T3 helps when the memory mixes domains (mixed streams; ImageNet-C and R/A/Sketch, where the sequential memory stays mixed, see below) and does nothing or hurts when the memory is one domain. The direct test is T3 with a reset at each domain start, vs Ramen-reset. That is **post-hoc** and was not run.

**The fp16 priority bug matters in sequential streams** (amendment; replayed memory composition, seed 0):
- DomainNet: the fp16 memory freezes during painting. Only 6.7% of quickdraw and 6.8% of real images ever enter it, and 0.3% of sketch; at the end it is 77% infograph + painting.
- Fixing it is worth **+0.22 ± 0.01** for Ramen (5/5 above the floor) and +0.17 for T3; quickdraw alone goes from +3.30 to +4.27 over NoAdapt.
- ImageNet-C and R/A/Sketch: no effect (seed 0: −0.00 and +0.00), as predicted, because their streams barely pass the 65,520 point.
- Practical consequence: any non-stationary-stream claim about Ramen should use fp32 priorities, and published Ramen numbers on long sequential streams are partly measuring this bug.

**Carrying memory across domains hurts Ramen everywhere.** Ramen-reset, which knows the boundaries, beats sequential Ramen:
- ImageNet-C **+0.82 ± 0.05**: 0/5 seeds favour sequential.
- DomainNet +0.27, or about +0.04 against fp32 Ramen (56.96 vs 56.92).
- R/A/Sketch +0.16, below the floor.

The ImageNet-C case is a capacity effect, not a bug. Its 75 slots × 1,000 classes exceed the 5,000 images per corruption, so nothing is evicted and old corruptions pile up. At the end of corruption n only about 1/n of the memory is the current one (9.5% at jpeg_compression). The largest per-domain losses from not resetting are on late corruptions: elastic_transform −3.5, frost −2.1, glass_blur −1.8, jpeg_compression −1.6.

**Order effect on Ramen (secondary 1).** Sequential vs mixed (each minus NoAdapt): ImageNet-C +5.89 vs +5.74, R/A/Sketch +1.63 vs +1.48, DomainNet +1.32 vs +1.44 (fp32 +1.54). Apart from the fp16 freeze, Ramen does about as well, or slightly better, when domains arrive in order.

**Adaptation lag (secondary 3).** In the first 10 batches of a domain the gain over NoAdapt is lower than later: ImageNet-C Ramen +4.50 vs +6.23, DomainNet +0.87 vs +1.47. Ramen-reset starts from an empty memory, yet on ImageNet-C it is *better* early (+5.79). The stale memory costs more than an empty one. On DomainNet it is slightly worse early (+0.69) and better later (+1.85). R/A/Sketch shows no lag.

**Caveats.** One domain order per dataset. The fp32 screens on ImageNet-C and R/A/Sketch are seed 0 only. The "T3 needs a mixed memory" reading is a hypothesis from per-domain patterns.

**What this changes.**
1. T3 is confirmed on three mixed streams and two of three sequential ones; it is not a universal improvement. DomainNet, the largest stream and the one with the largest memory (300 per class, top-10, β = 5), is where it fails.
2. Two findings about Ramen that matter for any continual-TTA claim: the fp16 priority freeze, and memory staleness across domains when capacity exceeds the per-domain volume (ImageNet-C −0.82). A boundary-free fix for staleness (e.g. evicting by age, or a capacity matched to the stream rate) is a candidate method change. Its oracle ceiling, from Ramen-reset, is +0.82 on ImageNet-C and +0.27 on DomainNet.
