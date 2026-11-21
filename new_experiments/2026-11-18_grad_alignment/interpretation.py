"""Interpretation text per dataset for the notebooks (kept apart so README.md and the notebooks say the same thing).

    from interpretation import INTERPRETATION
    INTERPRETATION['DomainNet']   # markdown, section 7 of 01_grad_alignment_domainnet.ipynb
"""

INTERPRETATION = {
    'DomainNet': r"""## 7. Interpretation (DomainNet, seed 0, 596,006 images)

**In one line.** Ramen's update is not a "true-class" direction. Where zero-shot is wrong, the memory turns the
image's own entropy gradient, which points *away* from the true class (median cosine −0.36), into a nearly
neutral direction (−0.035). That is no better than chance and below the shuffled control (+0.067). The images Ramen
fixes are the minority whose averaged update happens to tilt toward the true class (median +0.12). That tilt comes
almost entirely from the other 340-odd queues, not from the true class's own queue.

**What the numbers show.**
- **Zero-shot right (314,840 images, 52.8%).** The own entropy gradient is almost the CE gradient (median cosine
  0.99): minimising entropy *is* minimising cross-entropy toward the predicted label, and here that label is right.
  Ramen's averaged update keeps only part of that direction (median 0.23; positive on 269,951 of 314,840 = 85.7%).
  It still raises p(true class) on most images, so the dilution costs little: 10,302 breaks (3.3% of these images).
- **Zero-shot wrong (281,166 images, 47.2%).** Own entropy gradient: median −0.36, i.e. it would push further toward
  the wrong class. Ramen's update: median −0.035, positive on 122,924 of 281,166 (43.7%). Its sign step agrees with
  the oracle on 49.6% of the oracle's gradient mass, which is chance (50%). **The memory removes the harmful
  self-direction but does not replace it with the true-class direction** (H2 holds only in this weak sense).
- **Control.** Any image's update has cosine +0.07 with *another* image's CE gradient, a direction all images
  share. Measured against this, Ramen is above the control on zero-shot-right images (+0.23), on fixed images (+0.12)
  and over all images (+0.106 vs +0.068, H1 holds). On zero-shot-wrong images it is *below* the control (−0.035).
- **Fixed (20,428) vs broken (10,302).** Fixed images get +0.12 (75.2% positive). Broken images had a good own
  direction (+0.71) and Ramen replaced it with a neutral one (+0.013). H3b expected broken images to be *negative*.
  They are at about zero (+0.003 for the sign step), so H3b fails: Ramen breaks images by losing the right
  direction, not by reversing it.
- **Alignment decides the outcome (H3a).** Among zero-shot errors, the fix rate rises from 0.0% (3 of 28,117) in the
  lowest decile of cos(sign step, g_ce) to 32.6% (9,154 of 28,117) in the highest. Among zero-shot hits, the break
  rate falls from 17.4% (5,490 of 31,484) to 0.0% (10 of 31,484). This is partly expected: to first order the
  change of the true-label loss is proportional to this cosine. It confirms that one step with η = 0.01 is close to
  linear, so **any change that raises this cosine should convert directly into fixes.**
- **Where the alignment comes from (section 5.1).** Fixed images: +0.130 in total, of which other queues +0.116,
  the true-class queue +0.013, own entry +0.002 and the zero-shot queue −0.002. Correctly filed same-class
  neighbours contribute almost nothing. This matches the label-value oracle (2026-11-01: the image's own class gets
  ~2.5% of the update weight). The equal average over ~345 queues swamps the one queue that holds the right class.
- **Domains** behave alike. Zero-shot-wrong medians are −0.05 (quickdraw) to −0.01 (painting); quickdraw also has
  86.7% of its images wrong at zero-shot.
- **Over the stream.** On zero-shot-right images the alignment falls from ~0.43 to ~0.20 as more queues fill (each
  queue's share is 1/|C|). It drops again once the fp16 priority counter saturates (from position ~164k). Only
  168,423 of 596,006 images (28.3%) get their own entry stored. On zero-shot-wrong images it creeps from −0.09
  to −0.04.

**What this does not show.**
- One seed and one dataset. ImageNet-C (notebook 02) and EuroSAT (03) are needed before generalising. EuroSAT is
  where label values gave +26.70, so it is the most likely place to see a different picture.
- Cosines in a 39,936-dimensional space are small in absolute terms (random directions: about ±0.005), so +0.05
  is far from chance. The shuffled control (+0.07), not zero, is the fair baseline for "image-specific".
- The first-order link between cosine and outcome is a measurement, not a cause. It says what a better
  direction would buy; it does not say how to obtain one without labels.
- Zero-shot = the prediction Ramen filed each image under. It differs from the cached NoAdapt run on 1,050 of
  596,006 images (0.18%, fp16 near-ties).

**Next step.** Run notebooks 02 (ImageNet-C) and 03 (EuroSAT). If the pattern holds (neutralising, not
redirecting), a label-free change should raise cos(sign ĝ, g_ce) on zero-shot-wrong images *without* lowering it on
zero-shot-right ones. Examples: give the true-class queue more weight than an equal 1/|C| share, or use a
retrieval that is not an average over all queues. The global top-k oracle (+2.86 on DomainNet) points the same way.
Section 5.2 can then be reused to check, before any accuracy run, whether a candidate moves the cosine.
""",

    'ImageNetC5K': r"""## 7. Interpretation (ImageNet-C 5K, seed 0, 75,000 images)

**In one line.** Same picture as DomainNet, weaker and more uniform. Where zero-shot is wrong, the memory turns a
mildly wrong own direction (median −0.11) into a mildly positive one (+0.038). That is still slightly below the
shuffled control (+0.043). Practically all of the alignment comes from "other queues": with 1,000 classes and
$k = 1$, the true-class queue contributes one entry out of ~974.

**What the numbers show.**
- **Zero-shot right (20,460 images, 27.3%).** Own entropy gradient 0.96, Ramen's update 0.19 (positive on 17,914 of
  20,460 = 87.6%). Breaks: 2,860 (14.0% of these images).
- **Zero-shot wrong (54,540 images, 72.7%).** Own entropy gradient −0.11, Ramen's update +0.038 (positive on 31,778
  of 54,540 = 58.3%), sign agreement 52.9%, slightly above chance. H2 holds. The update is not above the control
  (+0.043), so it is no more about *this* image's true class than about another image's.
- **Fixed (7,209) vs broken (2,860).** Unlike DomainNet and EuroSAT, fixed images already had a *positive* own
  direction (+0.27): many ImageNet-C fixes are images whose true class was close behind the zero-shot one. Ramen
  keeps part of it (+0.16). Broken images lose a good own direction (+0.82 → +0.11). Their sign step is +0.023,
  not negative, so H3b fails as on DomainNet.
- **Alignment decides the outcome (H3a).** The fix rate rises from 0.3% (19 of 5,454) in the lowest decile of
  cos(sign step, g_ce) to 35.5% (1,936 of 5,454) in the highest. The break rate falls from 42.2% (863 of 2,046) to
  5.6% (115 of 2,046).
- **Where the alignment comes from.** Fixed images: +0.164 in total, of which other queues +0.162, true-class queue
  +0.001, own entry +0.000. The zero-shot queue part is exactly 0. With $k = 1$ an image's own entry is the
  nearest in its zero-shot queue (distance 0), so no other entry of that queue is retrieved. The own entry gets
  weight 1/974 and adds nothing.
- **Corruptions.** The zero-shot-wrong median ranges from −0.04 (elastic_transform) to +0.15 (gaussian_noise). The
  three noise corruptions are the only ones clearly above the control, and they are also where 83–85% of images
  are wrong at zero-shot.
- **Over the stream.** On zero-shot-wrong images the alignment rises from about 0.00 to about +0.05, crossing the
  control at about 40,000 images. Each photo appears 15 times (once per corruption). Later copies may therefore
  find an earlier copy of the *same photo* in another queue, which carries that photo's information. This fits
  the same-photo leak seen in the label-value oracle (2026-11-02), but this run did not log which entries were
  retrieved, so it is a **hypothesis**. The memory does not freeze: own entry stored for 74,465 of 75,000 images.

**What this does not show.**
- One seed. Cosines are small in absolute terms (random directions: about ±0.005).
- Whether the late rise on zero-shot-wrong images is a same-photo effect. Testing it needs the retrieved entries'
  photo ids, or a run with `exclude_groups`.
- Zero-shot = the prediction Ramen filed each image under. It differs from the cached NoAdapt run on 260 of 75,000
  images (0.35%, fp16 near-ties).

**Next step.** Compare across datasets (README): the true-class queue's share shrinks with the number of classes.
""",

    'EuroSAT': r"""## 7. Interpretation (EuroSAT, seed 0, 27,000 images)

**In one line.** With only 10 classes, Ramen's memory *does* point toward the true class. On zero-shot-wrong images
the update goes from −0.24 (own entropy gradient) to +0.10. The images Ramen fixes get their alignment mainly from
the true-class queue (+0.18 of +0.26). It is still below the shuffled control (+0.15) on zero-shot-wrong images as a
whole.

**What the numbers show.**
- **Zero-shot right (12,984 images, 48.1%).** Own entropy gradient 0.97, Ramen's update 0.49 (positive on 12,645 of
  12,984 = 97.4%), far above the control (+0.16). Breaks: 1,082 (8.3% of these images).
- **Zero-shot wrong (14,016 images, 51.9%).** Own entropy gradient −0.24, Ramen's update +0.10 (positive on 8,749
  of 14,016 = 62.4%). The sign step agrees with the oracle on 55.6% of its gradient mass, the clearest above-chance
  value of the three datasets. H2 holds. The update stays below the control (+0.148). The control is high on
  EuroSAT because with 10 classes another image's CE gradient often shares this image's class or the common
  "sharpen" direction.
- **Fixed (5,008) vs broken (1,082).** Fixed images had a *wrong* own direction (−0.21) that Ramen *reversed*
  (+0.27; positive on 4,373 of 5,008 = 87.3%). That is real redirection, which DomainNet and ImageNet-C did not
  show. Broken images lose part of a good own direction (+0.75 → +0.20) but stay positive (sign step +0.068), so
  H3b fails here too.
- **Alignment decides the outcome (H3a).** The fix rate rises from 3.5% (49 of 1,402) in the lowest decile to 76.8%
  (1,077 of 1,402) in the highest. The break rate falls from 44.5% (578 of 1,299) to 0.0% (0 of 1,299).
- **Where the alignment comes from.** Fixed images: +0.257 in total, of which true-class queue +0.181, other queues
  +0.106, zero-shot queue −0.023, own entry −0.007. Over all zero-shot-wrong images: true-class queue +0.109 of
  +0.081 in total. With 10 queues, the true class gets 1/10 of the update instead of 1/345 (DomainNet) or 1/974
  (ImageNet-C). Correctly filed same-class images then pull the update toward the true class. This fits the
  label-value oracle's large EuroSAT gain (+26.70).
- **Over the stream.** Flat: about +0.47 on zero-shot-right images and +0.05 → +0.08 on zero-shot-wrong images.
  The memory never fills (7,500 per queue) and never freezes: own entry stored for all 27,000 images.

**What this does not show.**
- One seed. EuroSAT is the noisiest dataset in the project (Ramen's noise floor is several points), so the size of
  each number is less certain than on the large datasets.
- Why fixes still need above-control alignment. On zero-shot-wrong images as a whole the update is below the
  control, so the true-class pull is enough only on the 36% of errors that Ramen fixes.
- Zero-shot = the prediction Ramen filed each image under. It differs from the cached NoAdapt run on 93 of 27,000
  images (0.34%, fp16 near-ties).

**Next step.** Compare across datasets (README). Weighting the true-class queue more works on EuroSAT because of
its few classes; the open question is how to find that queue without labels on 345 or 1,000 classes.
""",
}
