"""
Class-conditional mixture models over CLIP image embeddings, fitted by EM without labels.

One mixture component per class. EM starts from responsibilities R0 (the zero-shot posterior, i.e. the E-step of a
von Mises-Fisher mixture whose means are the class text embeddings and whose concentration is CLIP's logit scale 100),
then alternates M-step (re-estimate the class model from the soft assignments) and E-step (new posterior).

    from gmm import EM, SphericalVMF, SharedCovGaussian, UniformPrior
    em = EM(SphericalVMF(kappa=100.0), UniformPrior(), n_iter=10)
    trace = em.fit(Z, R0)          # Z: (N, d) unit-norm embeddings, R0: (N, C) rows sum to 1
    trace.preds[-1]                # argmax posterior after the last iteration; trace.preds[0] = argmax R0

Models (one job each): `SphericalVMF` (equal concentration, unit-norm means), `SharedCovGaussian` (LDA-style: class
means + one shared covariance, shrunk towards a scaled identity). Priors: `UniformPrior` (fixed 1/C) or `LearnedPrior`
(pi_c = N_c / N). No labels are used anywhere in this file; an oracle passes one-hot true labels as R0 (see
gmm_configs.py), which is the caller's choice.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

CHUNK = 100_000     # rows per block when forming (N, C) logits, to bound peak memory


def softmax_rows(logits):
    """Row softmax, in place on a float64 copy; returns (probs, log-sum-exp per row)."""
    m = logits.max(1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(1, keepdims=True)
    return e / s, (m + np.log(s))[:, 0]


# --------------------------------------------------------------------------- class priors
class UniformPrior:
    """pi_c = 1/C, never updated (zero-shot's implicit prior)."""

    def log_prior(self, R):
        return np.full(R.shape[1], -np.log(R.shape[1]))


class LearnedPrior:
    """pi_c = N_c / N, the standard EM update. Floor keeps log finite for classes that empty out."""

    def __init__(self, floor=1e-12):
        self.floor = floor

    def log_prior(self, R):
        pi = R.sum(0) / R.shape[0]
        return np.log(np.maximum(pi, self.floor))


# --------------------------------------------------------------------------- shared M-step statistics
def soft_class_sums(Z, R):
    """N_c = sum_i r_ic and S_c = sum_i r_ic z_i, accumulated in float64 block by block."""
    Nc = np.zeros(R.shape[1])
    Sc = np.zeros((R.shape[1], Z.shape[1]))
    for s in range(0, len(Z), CHUNK):
        r, z = R[s:s + CHUNK].astype(np.float64), Z[s:s + CHUNK].astype(np.float64)
        Nc += r.sum(0)
        Sc += r.T @ z
    return Nc, Sc


# --------------------------------------------------------------------------- models
class SphericalVMF:
    """
    log p(z | c) = log C_d(kappa) + kappa * mu_c . z, with ||mu_c|| = 1 and one kappa for all classes.
    M-step: mu_c = normalize(sum_i r_ic z_i). kappa is fixed, so C_d(kappa) is a constant that cancels in the
    posterior and is the same at every iteration (log_likelihood omits it).
    """

    def __init__(self, kappa=100.0, min_mass=1e-8):
        self.kappa, self.min_mass = kappa, min_mass
        self.means = None

    def init_means(self, means):
        self.means = means / np.linalg.norm(means, axis=1, keepdims=True)

    def m_step(self, Z, R):
        Nc, Sc = soft_class_sums(Z, R)
        new = Sc / np.maximum(np.linalg.norm(Sc, axis=1, keepdims=True), 1e-300)
        keep = Nc < self.min_mass          # a class with no mass keeps its previous mean
        if self.means is not None:
            new[keep] = self.means[keep]
        self.means = new
        return Nc

    def log_density(self, Z):
        """(N, C) log p(z_i | c) up to the shared constant log C_d(kappa)."""
        return self.kappa * (Z.astype(np.float64) @ self.means.T)


class SharedCovGaussian:
    """
    log p(z | c) = log N(z; mu_c, Sigma), one Sigma for all classes (linear discriminant).
    M-step: mu_c = S_c / N_c,  S = (Z^T Z - sum_c S_c S_c^T / N_c) / N  (pooled soft within-class scatter),
    Sigma = (1 - s) S + s (tr S / d) I  (shrinkage s; s = 0 is the exact maximum-likelihood update).
    """

    def __init__(self, shrinkage=0.1, min_mass=1e-8):
        self.shrinkage, self.min_mass = shrinkage, min_mass
        self.means = self.cov = None
        self._chol = self._logdet = None

    def init_means(self, means):
        self.means = np.asarray(means, np.float64)

    def m_step(self, Z, R):
        Nc, Sc = soft_class_sums(Z, R)
        ok = Nc >= self.min_mass
        new = np.array(self.means if self.means is not None else np.zeros_like(Sc))
        new[ok] = Sc[ok] / Nc[ok, None]
        self.means = new
        ZtZ = np.zeros((Z.shape[1], Z.shape[1]))
        for s in range(0, len(Z), CHUNK):
            z = Z[s:s + CHUNK].astype(np.float64)
            ZtZ += z.T @ z
        S = (ZtZ - (Sc[ok].T / Nc[ok]) @ Sc[ok]) / len(Z)
        S = (S + S.T) / 2
        d = S.shape[0]
        self.cov = (1 - self.shrinkage) * S + self.shrinkage * (np.trace(S) / d) * np.eye(d)
        self._chol = cho_factor(self.cov, lower=True)
        self._logdet = 2 * np.log(np.diag(self._chol[0])).sum()
        return Nc

    def log_density(self, Z):
        """(N, C) exact log N(z_i; mu_c, Sigma)."""
        z = Z.astype(np.float64)
        Pmu = cho_solve(self._chol, self.means.T)              # Sigma^{-1} mu_c, (d, C)
        lin = z @ Pmu                                           # z^T Sigma^{-1} mu_c
        quad_mu = 0.5 * np.einsum('dc,dc->c', self.means.T, Pmu)
        quad_z = 0.5 * np.einsum('nd,dn->n', z, cho_solve(self._chol, z.T))
        const = -0.5 * (z.shape[1] * np.log(2 * np.pi) + self._logdet)
        return lin - quad_mu[None, :] - quad_z[:, None] + const


# --------------------------------------------------------------------------- EM driver
@dataclass
class EMTrace:
    """Per-iteration record. Index 0 is the starting point R0 (no model fitted yet)."""
    preds: list = field(default_factory=list)          # argmax posterior, (N,) int32 per iteration
    log_likelihood: list = field(default_factory=list)  # sum_i log sum_c pi_c p(z_i|c) (model constant omitted)
    log_prior: list = field(default_factory=list)       # (C,) log pi used in that E-step
    mass: list = field(default_factory=list)            # (C,) N_c from that iteration's M-step
    final_probs: np.ndarray = None


class EM:
    """Alternates model.m_step(Z, R) and the E-step R = softmax(log p(z|c) + log pi) for n_iter rounds."""

    def __init__(self, model, prior, n_iter=10):
        self.model, self.prior, self.n_iter = model, prior, n_iter

    def e_step(self, Z, log_pi):
        R = np.empty((len(Z), len(log_pi)), np.float32)
        ll = 0.0
        for s in range(0, len(Z), CHUNK):
            probs, lse = softmax_rows(self.model.log_density(Z[s:s + CHUNK]) + log_pi[None, :])
            R[s:s + CHUNK] = probs
            ll += lse.sum()
        return R, ll

    def fit(self, Z, R0, callback=None):
        trace = EMTrace()
        R = np.asarray(R0, np.float32)
        trace.preds.append(R.argmax(1).astype(np.int32))
        trace.log_likelihood.append(np.nan)
        trace.log_prior.append(np.full(R.shape[1], np.nan))
        trace.mass.append(R.sum(0))
        for t in range(1, self.n_iter + 1):
            mass = self.model.m_step(Z, R)
            log_pi = self.prior.log_prior(R)
            R, ll = self.e_step(Z, log_pi)
            trace.preds.append(R.argmax(1).astype(np.int32))
            trace.log_likelihood.append(ll)
            trace.log_prior.append(log_pi)
            trace.mass.append(mass)
            if callback is not None:
                callback(t, trace)
        trace.final_probs = R
        return trace
