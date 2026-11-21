"""
Correctness tests for gmm.py and analysis.py, on synthetic data (no cached results, no GPU). Standard library unittest,
because pytest is not installed in the RAMEN env.

    ~/miniconda3/envs/RAMEN/bin/python -m unittest -v test_gmm

Each test checks one claim the experiment relies on, against an independent reference where one exists
(scipy for densities and softmax, sklearn for the supervised LDA posterior, brute-force loops for the M-step).
"""
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.special import softmax as scipy_softmax
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import gmm
from analysis import LabelScore, component_alignment, domain_average
from gmm import EM, LearnedPrior, SharedCovGaussian, SphericalVMF, UniformPrior, soft_class_sums, softmax_rows


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sphere_mixture(rng, n_per=400, C=5, d=16, spread=0.35, prior=None):
    """Points around C random unit directions, renormalised to the sphere (CLIP embeddings are unit norm)."""
    mu = unit(rng.normal(size=(C, d)))
    counts = np.full(C, n_per) if prior is None else rng.multinomial(n_per * C, prior)
    y = np.repeat(np.arange(C), counts)
    Z = unit(mu[y] + spread * rng.normal(size=(len(y), d)) / np.sqrt(d) * 3)
    return Z.astype(np.float32), y, mu


def soft_start(Z, means, kappa=100.0):
    return softmax_rows(kappa * Z.astype(np.float64) @ unit(means).T)[0].astype(np.float32)


class TestPrimitives(unittest.TestCase):
    def test_softmax_rows_matches_scipy(self):
        L = np.random.default_rng(0).normal(scale=30, size=(50, 7))
        P, lse = softmax_rows(L)
        np.testing.assert_allclose(P, scipy_softmax(L, axis=1), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(lse, logsumexp(L, axis=1), rtol=1e-12)

    def test_soft_class_sums_brute_force_across_chunks(self):
        rng = np.random.default_rng(1)
        Z, R = rng.normal(size=(23, 4)).astype(np.float32), rng.dirichlet(np.ones(3), size=23).astype(np.float32)
        old, gmm.CHUNK = gmm.CHUNK, 5           # force several blocks, including a short last one
        try:
            Nc, Sc = soft_class_sums(Z, R)
        finally:
            gmm.CHUNK = old
        Nc_ref = np.array([sum(R[i, c] for i in range(23)) for c in range(3)], dtype=np.float64)
        Sc_ref = np.array([sum(R[i, c] * Z[i].astype(np.float64) for i in range(23)) for c in range(3)])
        np.testing.assert_allclose(Nc, Nc_ref, rtol=1e-6)
        np.testing.assert_allclose(Sc, Sc_ref, rtol=1e-6, atol=1e-6)


class TestVMF(unittest.TestCase):
    def test_estep_with_text_means_is_clip_zero_shot_posterior(self):
        """Iteration-0 identity: vMF E-step, means = text embeddings, kappa = 100, uniform prior == CLIP softmax."""
        rng = np.random.default_rng(2)
        Z, W = unit(rng.normal(size=(200, 32))).astype(np.float32), unit(rng.normal(size=(9, 32)))
        m = SphericalVMF(kappa=100.0)
        m.init_means(W)
        R, _ = EM(m, UniformPrior()).e_step(Z, UniformPrior().log_prior(np.ones((1, 9))))
        np.testing.assert_allclose(R, scipy_softmax(100 * Z.astype(np.float64) @ W.T, axis=1), atol=1e-6)

    def test_mstep_is_normalised_soft_mean(self):
        rng = np.random.default_rng(3)
        Z, R = unit(rng.normal(size=(60, 8))).astype(np.float32), rng.dirichlet(np.ones(4), 60).astype(np.float32)
        m = SphericalVMF()
        m.m_step(Z, R)
        ref = unit(np.stack([(R[:, c:c + 1] * Z).sum(0) for c in range(4)]).astype(np.float64))
        np.testing.assert_allclose(m.means, ref, atol=1e-6)

    def test_empty_class_keeps_previous_mean(self):
        rng = np.random.default_rng(4)
        Z = unit(rng.normal(size=(30, 8))).astype(np.float32)
        R = np.zeros((30, 3), np.float32)
        R[:, :2] = rng.dirichlet(np.ones(2), 30)       # class 2 gets no mass
        m = SphericalVMF()
        W = unit(rng.normal(size=(3, 8)))
        m.init_means(W)
        m.m_step(Z, R)
        np.testing.assert_allclose(m.means[2], W[2])


class TestSharedCovGaussian(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.Z = rng.normal(size=(300, 6)).astype(np.float32)
        self.R = rng.dirichlet(np.ones(4), 300).astype(np.float32)

    def test_mstep_matches_brute_force_soft_scatter(self):
        m = SharedCovGaussian(shrinkage=0.0)
        m.m_step(self.Z, self.R)
        Z = self.Z.astype(np.float64)
        mu = np.stack([(self.R[:, c:c + 1] * Z).sum(0) / self.R[:, c].sum() for c in range(4)])
        S = sum(self.R[i, c] * np.outer(Z[i] - mu[c], Z[i] - mu[c]) for i in range(300) for c in range(4)) / 300
        np.testing.assert_allclose(m.means, mu, atol=1e-6)
        np.testing.assert_allclose(m.cov, S, atol=1e-6)

    def test_shrinkage_formula(self):
        a, b = SharedCovGaussian(shrinkage=0.0), SharedCovGaussian(shrinkage=0.3)
        a.m_step(self.Z, self.R)
        b.m_step(self.Z, self.R)
        d = a.cov.shape[0]
        np.testing.assert_allclose(b.cov, 0.7 * a.cov + 0.3 * np.trace(a.cov) / d * np.eye(d), atol=1e-10)

    def test_log_density_matches_scipy(self):
        m = SharedCovGaussian(shrinkage=0.1)
        m.m_step(self.Z, self.R)
        L = m.log_density(self.Z)
        ref = np.stack([multivariate_normal(m.means[c], m.cov).logpdf(self.Z.astype(np.float64))
                        for c in range(4)], 1)
        np.testing.assert_allclose(L, ref, rtol=1e-9, atol=1e-9)

    def test_oracle_one_hot_fit_equals_sklearn_lda(self):
        """Supervised fit (one-hot R0, shrinkage 0, uniform prior) == sklearn LinearDiscriminantAnalysis posterior."""
        Z, y, _ = sphere_mixture(np.random.default_rng(6), n_per=300, C=4, d=10)
        R0 = np.eye(4, dtype=np.float32)[y]
        trace = EM(SharedCovGaussian(shrinkage=0.0), UniformPrior(), n_iter=1).fit(Z, R0)
        sk = LinearDiscriminantAnalysis(solver='lsqr', priors=np.full(4, 0.25)).fit(Z.astype(np.float64), y)
        np.testing.assert_allclose(trace.final_probs, sk.predict_proba(Z.astype(np.float64)), atol=1e-5)
        np.testing.assert_array_equal(trace.preds[1], sk.predict(Z.astype(np.float64)))


class TestEM(unittest.TestCase):
    def _monotone(self, model, prior):
        Z, _, mu = sphere_mixture(np.random.default_rng(7), C=5, d=12)
        noisy_text = unit(mu + 0.8 * np.random.default_rng(8).normal(size=mu.shape))
        trace = EM(model, prior, n_iter=15).fit(Z, soft_start(Z, noisy_text, kappa=20.0))
        ll = np.array(trace.log_likelihood[1:])
        self.assertTrue(np.all(np.diff(ll) >= -1e-6 * np.abs(ll[:-1])), f'log-likelihood decreased: {ll}')

    def test_monotone_vmf_uniform(self):
        self._monotone(SphericalVMF(kappa=20.0), UniformPrior())

    def test_monotone_vmf_learned_prior(self):
        self._monotone(SphericalVMF(kappa=20.0), LearnedPrior())

    def test_monotone_lda_uniform(self):
        self._monotone(SharedCovGaussian(shrinkage=0.0), UniformPrior())

    def test_monotone_lda_learned_prior(self):
        self._monotone(SharedCovGaussian(shrinkage=0.0), LearnedPrior())

    def test_recovers_planted_mixture_from_poor_start(self):
        """When the class structure is in the data and each start mean is nearest its own cluster, EM from a poor
        text-like start must find it (so a null result on real data is about the data, not a code path that cannot
        improve)."""
        rng = np.random.default_rng(9)
        Z, y, mu = sphere_mixture(rng, n_per=500, C=6, d=16, spread=0.25)
        noisy_text = unit(0.55 * mu + 0.45 * np.roll(mu, -1, axis=0))    # each start mean biased towards the next class
        np.testing.assert_array_equal((noisy_text @ mu.T).argmax(1), np.arange(6))   # precondition
        R0 = soft_start(Z, noisy_text, kappa=20.0)
        acc0 = (R0.argmax(1) == y).mean()
        self.assertLess(acc0, 0.9)
        for model in (SphericalVMF(kappa=20.0), SharedCovGaussian(shrinkage=0.1)):
            trace = EM(model, UniformPrior(), n_iter=20).fit(Z, R0)
            acc = (trace.preds[-1] == y).mean()
            self.assertGreater(acc, 0.97, f'{type(model).__name__}: {acc0:.3f} -> {acc:.3f}')

    def test_label_switching_when_start_names_clusters_wrongly(self):
        """Documented limitation, not a bug: if start means sit nearest the wrong cluster, EM finds the clusters
        but keeps the wrong class names (accuracy stays low, best-permutation accuracy is much higher)."""
        from scipy.optimize import linear_sum_assignment
        rng = np.random.default_rng(9)
        Z, y, mu = sphere_mixture(rng, n_per=500, C=6, d=16, spread=0.25)
        noisy_text = unit(mu + 0.9 * rng.normal(size=mu.shape))
        self.assertFalse(np.array_equal((noisy_text @ mu.T).argmax(1), np.arange(6)))   # start misnames clusters
        p = EM(SphericalVMF(20.0), UniformPrior(), n_iter=20).fit(Z, soft_start(Z, noisy_text, 20.0)).preds[-1]
        M = np.zeros((6, 6))
        np.add.at(M, (y, p), 1)
        r, c = linear_sum_assignment(-M)
        acc, matched = (p == y).mean(), M[r, c].sum() / len(y)
        self.assertLess(acc, 0.6)
        self.assertGreater(matched, acc + 0.2)

    def test_chunking_does_not_change_result(self):
        Z, _, mu = sphere_mixture(np.random.default_rng(10), C=4, d=8)
        R0 = soft_start(Z, unit(mu + 0.5), kappa=20.0)
        outs = []
        old = gmm.CHUNK
        try:
            for chunk in (7, 10**6):
                gmm.CHUNK = chunk
                outs.append(EM(SharedCovGaussian(0.1), LearnedPrior(), n_iter=5).fit(Z, R0).final_probs)
        finally:
            gmm.CHUNK = old
        np.testing.assert_allclose(outs[0], outs[1], atol=1e-5)

    def test_row_permutation_permutes_output(self):
        """The fit is transductive and order-free: shuffling the stream only reorders the labels."""
        rng = np.random.default_rng(11)
        Z, _, mu = sphere_mixture(rng, C=4, d=8)
        R0 = soft_start(Z, unit(mu + 0.5), kappa=20.0)
        perm = rng.permutation(len(Z))
        a = EM(SphericalVMF(20.0), UniformPrior(), n_iter=5).fit(Z, R0).preds[-1]
        b = EM(SphericalVMF(20.0), UniformPrior(), n_iter=5).fit(Z[perm], R0[perm]).preds[-1]
        np.testing.assert_array_equal(a[perm], b)

    def test_zero_iterations_returns_start(self):
        Z, _, mu = sphere_mixture(np.random.default_rng(12), C=3, d=8)
        R0 = soft_start(Z, mu)
        tr = EM(SphericalVMF(), UniformPrior(), n_iter=0).fit(Z, R0)
        np.testing.assert_array_equal(tr.preds[0], R0.argmax(1))
        self.assertEqual(len(tr.preds), 1)


class TestLabelScore(unittest.TestCase):
    def test_hand_example(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        zs = np.array([0, 1, 1, 0, 2, 0])          # right: 0, 2, 4  -> 3
        new = np.array([0, 0, 2, 1, 2, 0])         # right: 0, 1, 3, 4 -> 4
        dom = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
        s = LabelScore.of(new, zs, labels, dom)
        self.assertEqual((s.fixed, s.broken, s.changed, s.zs_wrong, s.zs_right), (2, 1, 3, 3, 3))
        self.assertAlmostEqual(s.acc_pooled, 400 / 6)
        self.assertAlmostEqual(s.acc_domain_avg, 100 * (2 / 3 + 2 / 3) / 2)
        self.assertAlmostEqual(s.rescue_rate, 2 / 3)
        self.assertAlmostEqual(s.break_rate, 1 / 3)

    def test_component_alignment(self):
        labels, dom = np.repeat([0, 1, 2, 3], 50), np.tile(np.repeat(['a', 'b'], 25), 4)
        a = component_alignment(labels, labels, dom)          # components = classes
        self.assertAlmostEqual(a['nmi_class'], 1.0)
        self.assertAlmostEqual(a['nmi_domain'], 0.0, places=12)
        b = component_alignment((dom == 'b').astype(int), labels, dom)   # components = domains
        self.assertAlmostEqual(b['nmi_domain'], 1.0)
        self.assertAlmostEqual(b['nmi_class'], 0.0, places=12)

    def test_domain_average_is_unweighted_mean_of_domains(self):
        self.assertAlmostEqual(domain_average(np.array([1, 1, 1, 0]), np.array(['a', 'a', 'a', 'b'])), 50.0)

    def test_verdict_rule(self):
        s = LabelScore(n=100, acc_domain_avg=0, acc_pooled=62, zs_acc_domain_avg=0, zs_acc_pooled=50, zs_wrong=50,
                       zs_right=50, fixed=13, broken=1, changed=14, max_class_share=0.1, n_classes_used=10)
        self.assertEqual(s.verdict(0.5, 10, 0.05), 'pass')
        s.broken = 3                                 # 6% of zero-shot-right broken
        self.assertEqual(s.verdict(0.5, 10, 0.05), 'weak')
        s.acc_pooled = 50.2
        self.assertEqual(s.verdict(0.5, 10, 0.05), 'null')
        s.acc_pooled = 49
        self.assertEqual(s.verdict(0.5, 10, 0.05), 'hurts')


if __name__ == '__main__':
    unittest.main()
