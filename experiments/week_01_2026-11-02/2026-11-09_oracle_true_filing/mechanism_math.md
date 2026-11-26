# Why filing Ramen's memory by the true label lowers accuracy: the maths

Companion to [oracle_true_filing.ipynb](oracle_true_filing.ipynb), which has the measurements. This file has the derivation. Every number here is quoted from that notebook's outputs (seed 0), and the section is named each time. Paper references are to `docs/papers/pdf/ramen_2604.21728.pdf` (§3.2, Remark 4.3, App. B).

**What is exact and what is approximate.** Steps 1–3 are exact for the model in §1. Step 4 (what filing does to the active set) uses one stated approximation, the "dominant push" approximation. §1 lists the simplifications against real Ramen, and §9 lists what the derivation does not cover. `[hypothesis]` marks claims that the runs support but have not tested directly.

---

## 0. The argument in five lines

1. A memory entry $j$ stores $g_j=\nabla_\theta H_j$. That gradient makes the model more confident in $j$'s **zero-shot** class $\hat y_j$, wherever $j$ is filed. Call $\hat y_j$ its *push class*.
2. Ramen's update for a query decomposes exactly into two parts. The first is a **net-push term**, a class-prior shift proportional to the active set's average push vector $\bar\delta$. The second is a **covariance term**, a class-discriminative feature reweighting (the paper's Theorem 4.1).
3. With zero-shot filing, every queue $c$ pushes toward $c$, so the active set has equal counts per push class. Then $\bar\delta\approx 0$ and the covariance term is as large as it can be: this is the paper's *prediction balance*.
4. With true-label filing, queue $c$ holds true class $c$. On a corrupted domain, though, most of its same-domain entries are zero-shot-labelled as the domain's attractor $A$. So $\bar\delta$ points toward $A$ for **every** query of that domain, and the covariance term loses its class content.
5. The result is a prior shift toward $A$ added to every query. That flips correct predictions to $A$ and undoes Ramen's corrections. Where the attractor is strong enough (DomainNet quickdraw), predictions fall below zero-shot.

---

## 1. Setup and notation

Real Ramen (`src/methods/Ramen.py:111-168`), for a query $x_i$:

$$z_i=\frac{f_{\theta_0}(x_i)}{\lVert f_{\theta_0}(x_i)\rVert},\quad p_i=\mathrm{softmax}(\tau W^\top z_i),\quad \hat y_i=\arg\max_c p_{ic},\quad H_i=-\sum_c p_{ic}\log p_{ic},\quad g_i=\nabla_\theta H_i\big|_{\theta_0}$$

- **Write:** the entry $(z_i,g_i,H_i,t_i)$ goes into queue $q_i$. Ramen uses $q_i=\hat y_i$, and the ORACLE copy uses $q_i=y_i$.
- **Retrieve:** take the top-$k$ entries by distance from every non-empty queue. This is the active set $S_i$. Weights are $w_{ij}=e^{-H_j}e^{-\beta\lVert z_i-z_j\rVert}$.
- **Aggregate:** $G_i=\frac{1}{|\mathcal C^+|}\sum_{c}\sum_{j\in\mathrm{top}k_c}w_{ij}g_j$.
- **Step, predict, reset:** $\theta_i=\theta_0-\eta\,\mathrm{sign}(G_i)$.

**Normalised weights.** $\mathrm{sign}(\cdot)$ ignores any positive scale, so the step's direction is unchanged if we write
$$G_i\ \propto\ \sum_{j\in S_i}\alpha_{j}\,g_j,\qquad \alpha_j=\frac{w_{ij}}{\sum_{l\in S_i}w_{il}},\quad \sum_j\alpha_j=1 .$$

**The analysed model.** This follows the paper's App. B, extended from 2 to $C$ classes. Take one normalisation layer with affine parameters $w,b\in\mathbb R^d$ at $w=\mathbf 1$, $b=\mathbf 0$, a normalised feature $v_j$, the embedding $z_j=v_j\odot w+b$, text embeddings $T=[t_1,\dots,t_C]\in\mathbb R^{d\times C}$, and logits $l_{jc}=\tau\,t_c^\top z_j$.

Simplifications against real Ramen:
- Real Ramen has 26 LayerNorms deep in the network, not one at the output.
- Real Ramen L2-normalises $z$.
- Real Ramen steps with SignSGD, while the analysis below uses plain gradient descent (GD) first. §9 discusses what changes under SignSGD.

The paper makes the same single-layer, plain-GD simplifications in §4 and App. B.

---

## 2. Step 1: what one memory entry does (the push vector)

Define the **push vector** of entry $j$ as the negative entropy gradient with respect to its logits:
$$\delta_j:=-\frac{\partial H_j}{\partial l_j}\in\mathbb R^{C}.$$

**Lemma 1.** $\displaystyle \delta_{jc}=p_{jc}\left(\log p_{jc}+H_j\right)$.

*Proof.* We have $\partial p_{jk}/\partial l_{jc}=p_{jk}(\mathbb 1[k{=}c]-p_{jc})$ and $\partial H_j/\partial p_{jk}=-(\log p_{jk}+1)$. So
$$\frac{\partial H_j}{\partial l_{jc}}=-(\log p_{jc}+1)p_{jc}+p_{jc}\sum_k p_{jk}(\log p_{jk}+1)=-(\log p_{jc}+1)p_{jc}+p_{jc}(1-H_j)=-p_{jc}(\log p_{jc}+H_j).\ \square$$

**Properties.**
1. $\sum_c\delta_{jc}=\sum_c p_{jc}\log p_{jc}+H_j=0$. A push only *redistributes* logit mass.
2. $\delta_{jc}>0 \iff p_{jc}>e^{-H_j}$. The class $\hat y_j$ always qualifies, because $\log p_{j\hat y_j}\ge\sum_k p_{jk}\log p_{jk}=-H_j$. So **a GD step on $H_j$ always raises the logit of $\hat y_j$** relative to the classes with low probability. This is what "push class" means, and it doesn't depend on $y_j$ or on the queue $j$ sits in.
3. *Check against the paper (binary case).* With $C=2$: $\log p_{j1}+H_j=p_{j0}(\log p_{j1}-\log p_{j0})=p_{j0}(l_{j1}-l_{j0})$, so $\delta_{j1}=p_{j0}p_{j1}(l_{j1}-l_{j0})$. This is the negative of the paper's $\partial H_j/\partial l_{j1}$ (App. B).

**Dominant-push approximation** (used only in §5–§6). Write
$$\delta_j\approx s_j\,u_{\hat y_j},\qquad u_c:=e_c-\tfrac1C\mathbf 1,\qquad s_j\ge 0 .$$
This keeps the largest component, toward $\hat y_j$, and drops the smaller pushes toward the other high-probability classes. The exact $\delta_j$ also pushes toward any class with $p_{jc}>e^{-H_j}$. The approximation is crude for near-ties, but it is exactly the quantity the notebook logs: weight share per push class $\hat y_j$, in section 5.

---

## 3. Step 2: one entry's gradient on the affine parameters

By the chain rule, $\partial H_j/\partial z_j=\sum_c\frac{\partial H_j}{\partial l_{jc}}\tau t_c=-\tau\,T\delta_j$. Then:

$$\boxed{\ g^{b}_j=-\tau\,T\delta_j,\qquad g^{w}_j=-\tau\,(T\delta_j)\odot v_j\ }$$

Define the vector $a_j:=T\delta_j\in\mathbb R^d$. It is the text-space direction of $j$'s push: a mix of the class text embeddings, weighted by $\delta_j$. Under the dominant-push approximation, $a_j\approx s_j(t_{\hat y_j}-\bar t)$ with $\bar t=\frac1C\sum_c t_c$.

---

## 4. Step 3: the aggregate splits exactly into net push + covariance

Take the weighted means over the active set: $\bar\delta=\sum_j\alpha_j\delta_j$, $\bar a=\sum_j\alpha_j a_j=T\bar\delta$, and $\bar v=\sum_j\alpha_jv_j$. Since $\sum_j\alpha_j=1$, the weighted covariance identity gives

$$\sum_j\alpha_j\,a_j\odot v_j=\bar a\odot\bar v+\underbrace{\sum_j\alpha_j\,(a_j-\bar a)\odot(v_j-\bar v)}_{=:\ \mathrm{Cov}_\alpha(a,v)} .$$

So the GD update $\Delta=-\eta\sum_j\alpha_jg_j$ is **exactly**

$$\boxed{\ \Delta b=\eta\tau\,T\bar\delta,\qquad \Delta w=\eta\tau\Big[\underbrace{(T\bar\delta)\odot\bar v}_{\text{net push}}+\underbrace{\mathrm{Cov}_\alpha(a,v)}_{\text{covariance}}\Big]\ }$$

### What each part does to the query's logits

To first order, $\Delta l_{ic}=\tau\,t_c^\top(\Delta w\odot v_i+\Delta b)$.

**Net-push part (a class prior).** Let $K=T^\top T$ be the $C\times C$ Gram matrix of the text embeddings.
- From $\Delta b$: $\Delta l_{ic}=\eta\tau^2(K\bar\delta)_c$. Under the rough approximation $K\approx\kappa I+\gamma\mathbf 1\mathbf 1^\top$ (equal pairwise text similarity), and using $\mathbf 1^\top\bar\delta=0$:
$$\Delta l_{ic}\approx\eta\tau^2\kappa\,\bar\delta_c .$$
  This doesn't depend on the query's features. It is a **prior shift toward the classes the active set pushes toward**.
- From the $(T\bar\delta)\odot\bar v$ part of $\Delta w$: $\Delta l_{ic}=\eta\tau^2\,t_c^\top\mathrm{diag}(v_i\odot\bar v)\,T\bar\delta$. This is the same form with a feature-weighted Gram matrix. When retrieval is domain-consistent ($\bar v$ close to $v_i$'s domain), it is again roughly a prior shift proportional to $\bar\delta$.

**Covariance part (the discriminative reweighting).** Under the dominant-push approximation,
$$\mathrm{Cov}_\alpha(a,v)_h\approx\sum_j\alpha_j\big(s_j(t_{\hat y_j,h}-\bar t_h)-\bar a_h\big)(v_{jh}-\bar v_h).$$
Coordinate $h$ is amplified when the entries pushed toward class $c$ have image features that lean along $t_c$ in that coordinate. In other words, $h$ separates the pseudo-classes *locally*, among the query's same-domain neighbours. This is the multi-class form of the paper's $M=\mathbb E_{j\in S}[p_{j0}p_{j1}v_jv_j^\top]$ in Theorem 4.1 (Eq. 8–9) and Remark 4.2. It is the part of the update that does useful work.

**The paper's App. B in this notation.** In the binary case, the paper's bias gradient $\mathbb E_{j\in S}(l_{j1}-l_{j0})p_{j0}p_{j1}\,(t_1-t_0)$ equals $T\bar\delta$ up to sign. The paper's claim "class-balanced ⇒ $\mathbb E(\cdot)\approx0$ ⇒ bias gradient negligible" is the statement $\bar\delta\approx0$.

---

## 5. Step 4: what filing does to the net push $\bar\delta$

Under the dominant-push approximation,
$$\bar\delta\approx\sum_c m_c\,u_c=m-\bar m\,\mathbf 1,\qquad m_c:=\sum_{j\in S_i:\ \hat y_j=c}\alpha_js_j\quad(\text{push mass on class }c),\qquad \bar m=\tfrac1C\textstyle\sum_cm_c .$$

So **$\bar\delta=0$ exactly when push mass is spread evenly over classes**, and the prior shift on class $c$ is proportional to $m_c-\bar m$.

### Ramen: balanced by construction

Every entry of queue $c$ has $\hat y_j=c$, and $S_i$ takes $k$ entries from every non-empty queue. So the active set holds exactly $k$ entries per push class, and $m_c$ differs across classes only through the weights $\alpha_js_j$. With equal counts there is no class-level imbalance (paper §3.2, "prediction balance"). The residual imbalance comes from the weights. The nearest entries tend to sit in the query's own zero-shot queue, so the residual push leans toward $\hat y_i$, a mild self-training. Measured: 22.0% of active-set weight pushes toward the query's zero-shot class on CIFAR-10-C (notebook section 5).

### ORACLE: concentrated on the attractor

Queue $c$ now holds entries with true class $c$, and their push classes are their zero-shot labels. Let $\rho_{c\to a}$ be the rate at which, among the query's same-domain neighbours of true class $c$, zero-shot predicts $a$. Then
$$m_a\approx k\,\overline{\alpha s}\sum_{c}\rho_{c\to a}.$$

For the domain's attractor $A$, with roughly uniform classes, $\sum_c\rho_{c\to A}\approx C\cdot P(\hat y=A\mid\text{domain})$. The idealised share of push mass on $A$ is therefore

$$\boxed{\ \text{Ramen: }\ \frac{1}{|\mathcal C^+|}\qquad\text{vs.}\qquad\text{ORACLE: }\ P(\hat y=A\mid\text{domain})\ }$$

The ORACLE exceeds a balanced share by the factor $|\mathcal C^+|\cdot P(\hat y=A\mid\text{domain})$. That factor is large whenever zero-shot over-predicts $A$, which is exactly what an attractor is, and it grows with the number of classes.

There is a second, quieter effect. The query's *own* true-class queue $y_i$ is now filled with same-domain images of class $y_i$, and those are exactly the images zero-shot misreads as $A$ (notebook interpretation, point 2, and example 2). So the queue that "should" pull the query toward its true class also pushes toward $A$.

### Toy example (3 classes, one noisy domain, attractor = bird)

Zero-shot on this domain behaves as follows: true cat → 50% cat, 50% bird; true dog → 50% dog, 50% bird; true bird → 100% bird. Take $k$ entries per queue, with equal $\alpha s$.

| | queue cat | queue dog | queue bird | push mass $m$ (cat, dog, bird) | $\bar\delta\propto m-\bar m$ |
|---|---|---|---|---|---|
| **Ramen** (filed by $\hat y$) | $k$ push cat | $k$ push dog | $k$ push bird | $(k,\ k,\ k)$ | $(0,\ 0,\ 0)$ |
| **ORACLE** (filed by $y$) | $k/2$ cat, $k/2$ bird | $k/2$ dog, $k/2$ bird | $k$ bird | $(k/2,\ k/2,\ 2k)$ | $(-k/2,\ -k/2,\ +k)$ |

The ORACLE puts $2/3$ of the push on bird, against Ramen's $1/3$. That matches the formula: $P(\hat y=\text{bird})=(0.5+0.5+1)/3=2/3$. Every query from this domain then gets a prior shift toward bird, even a correctly classified cat.

---

## 6. Step 5: what filing does to the covariance term

Suppose every entry in the active set had the same push class $A$. Then $a_j=s_j b$ with a fixed vector $b=t_A-\bar t$, and
$$\mathrm{Cov}_\alpha(a,v)=b\odot\sum_j\alpha_j(s_j-\bar s)(v_j-\bar v).$$
This only rescales features along the attractor direction $b$, by how strongly each entry is pushed. It carries **no information about which features separate classes**. More generally, the between-class part of the covariance scales with the spread of push mass over classes. A heuristic measure of that spread is the impurity $1-\sum_c\pi_c^2$ of the push-mass shares $\pi_c=m_c/\sum m$. It is largest under Ramen's balanced filing, where $\pi$ is near uniform, and falls as push concentrates on $A$.

This is the paper's Remark 4.3, "prediction balance keeps the class-relevant features exhibiting large variance $M_{hh}$", made explicit: the variance that counts is the variance *across push classes*.

So true-label filing hurts twice. **The harmful term grows** ($\bar\delta$ points toward $A$), and **the useful term shrinks** (less push-class spread to reweight features with).

---

## 7. Step 6: when a prediction flips, and why DomainNet goes below zero-shot

Take a query with true class $y\neq A$ that Ramen classifies correctly. After the step, its logit margin over the attractor is, to first order,
$$\underbrace{(l_{iy}-l_{iA})}_{\text{zero-shot margin}}\;+\;\eta\tau^2\kappa\,\underbrace{(\bar\delta_{iy}-\bar\delta_{iA})}_{\text{net push}}\;+\;\underbrace{[\text{covariance term}]_{y}-[\cdot]_{A}}_{\text{discriminative}} .$$

- **Under Ramen**, the net push is about 0, so the discriminative term decides. Ramen's correction of zero-shot errors lives here.
- **Under the ORACLE**, $\bar\delta_{iA}-\bar\delta_{iy}$ is large and positive and the discriminative term is weaker. The prediction flips to $A$ whenever the prior shift beats the margin. The first to flip are images with small margins, which are exactly the zero-shot errors that Ramen had only just corrected.
- **Below zero-shot.** Zero-shot has no step, so its net push is 0. If the ORACLE's prior shift toward $A$ outweighs what is left of the discriminative term, it predicts $A$ *more often than zero-shot does*, and accuracy drops below zero-shot. The factor $|\mathcal C^+|\cdot P(\hat y=A\mid\text{domain})$ is largest when there are many classes and the attractor is strong. DomainNet quickdraw has 345 classes and "squiggle" absorbs a large share of zero-shot predictions there, so that is where it happens.

---

## 8. Checking the derivation against the measurements

All values are from [oracle_true_filing.ipynb](oracle_true_filing.ipynb), seed 0.

| Prediction from the maths | Measured (notebook section) | Holds? |
|---|---|---|
| The ORACLE's active set is more concentrated on one push class than Ramen's (§5) | Weight on the single most-pushed class, Ramen → ORACLE: CIFAR-10-C 22.6 → 38.7%, CIFAR-100-C 8.9 → 18.2%, ImageNet-C 2.2 → 6.0%, DomainNet 3.2 → 15.1% (section 5) | yes, all 4 |
| Predictions move back toward each domain's attractor (§7) | Share of a domain's images predicted as its attractor, zero-shot / Ramen / ORACLE: CIFAR-10-C 20.2 / 10.6 / 14.6; CIFAR-100-C 7.65 / 2.90 / 5.99; ImageNet-C 7.40 / 0.98 / 4.57; DomainNet 5.93 / 2.28 / **7.86** (section 3) | yes. DomainNet's ORACLE is above even zero-shot |
| Below zero-shot where the attractor is strongest (§7) | "squiggle" predicted: zero-shot 7.72%, Ramen 2.05%, ORACLE **11.09%** (true 0.29%); DomainNet accuracy: ORACLE 55.65 vs zero-shot 55.72 (sections 1 and 7) | yes |
| Broken images are mostly zero-shot errors that Ramen had fixed (§7) | Most broken images (64–87%) were zero-shot errors Ramen corrected; on CIFAR-10-C 58.1% repeat the zero-shot answer (section 5, second table) | yes |
| The prior is *relative*: more push toward the true class does not help if push toward $A$ grows more (§5) | CIFAR-10-C, all images: weight pushing toward the TRUE class, Ramen 21.1% → ORACLE 27.7%, yet accuracy falls 72.61 → 69.42 (sections 1 and 5) | consistent |
| The predicted-class histogram is further from the truth (§5) | TV(pred, true), Ramen → ORACLE: 5.46 → 7.26, 17.23 → 23.63, 25.25 → 27.47, 18.19 → 23.19 (section 3) | yes, all 4 |

**Magnitudes are not predicted.** The idealised share $P(\hat y=A\mid\text{domain})$ counts entries. The measured shares are weighted by $w_{ij}$, ignore $s_j$ (that is, $\lVert g_j\rVert$), and come from a per-query neighbourhood rather than the whole domain. The derivation predicts the *direction* of every row above, not the size.

---

## 9. What the derivation does not cover

1. **SignSGD.** Real Ramen steps with $-\eta\,\mathrm{sign}(G_i)$, not with $-\eta G_i$. For the bias coordinates, $G^b\propto T\bar\delta$, so they move by $\pm\eta$ however small $\bar\delta$ is. Under SignSGD, the paper's "the bias gradient becomes negligible" (App. B) is really "the bias step's **direction** is set by the residual imbalance". The argument above still goes through in direction terms. Under Ramen, the residual $\bar\delta$ is weak and query-specific, leaning toward the query's own zero-shot class. Under the ORACLE, it points toward the *same* attractor for every query of a domain. But the claim that the bias step is *small* under Ramen does not carry over to SignSGD. `[my observation; not tested]`
2. **Depth and normalisation.** The analysis covers one affine layer at the output. Real Ramen adapts 26 LayerNorms and L2-normalises $z$. The paper's theory makes the same simplification (§4).
3. **Why random filing hurts less than the ORACLE** (CIFAR-10-C 70.90 vs 69.42; CIFAR-100-C 43.24 vs 41.74; section 1). Random filing's active sets are even *more* concentrated (most-pushed class 54.4% on CIFAR-10-C, section 5), so concentration alone is not the full story. `[hypothesis]` A random queue is a random $1/C$ slice of the stream, so its top-$k$ entries are close to the query's overall nearest neighbours. Their push points mostly to the query's **own** zero-shot class: 50.4% of the weight on CIFAR-10-C, against the ORACLE's 34.9%. That makes $\bar\delta$ a self-training push, which keeps a correct zero-shot answer correct and fails only to fix wrong ones. The ORACLE's per-true-class top-$k$ instead forces in same-domain images of *every* class, whose push is the domain-wide confusion dominated by $A$. That direction is wrong even for queries that zero-shot gets right. Random filing is roughly the paper's "w/o PB" ablation (one queue, top $C\cdot k$; §5, Fig. 5), which also scores below Ramen. A direct test would split each query's push weight into "own zero-shot class" and "domain attractor".
4. **Why Ramen *removes* attractors** (attractor share 20.2% → 10.6% on CIFAR-10-C). That comes from the covariance term together with the residual self-training push. It is not derived here.
5. **One seed.** The accuracy gaps are about 6–17× each dataset's noise floor (notebook interpretation), but per-domain statements need their own spread.

---

## 10. What this means for the research direction

- Any change that only cleans up the memory's **labels** (filing, relabelling, purity) changes *which* entries are retrieved together, never what each entry pushes toward. By §5, cleaner labels break the balance of push classes, so this family has a negative ceiling under Ramen's update. That is the closed row in `docs/research_state.md`.
- The quantity that matters is $\bar\delta$, the push balance of the active set. Two levers follow:
  - (a) **Control the balance directly.** For example, cap each push class's share of the active-set weight at $1/|\mathcal C^+|$. This is the next-step proposal in the notebook's interpretation.
  - (b) **Change what each entry pushes toward**, that is, the stored value. That is the [oracle_value](../../week_02_2026-11-09/2026-11-11_oracle_value/) experiment, whose notebook is not finished yet.
