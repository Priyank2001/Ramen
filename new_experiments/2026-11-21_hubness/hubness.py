"""
Label-free, causal hubness corrections of CLIP's class scores (PREREGISTRATION.md), one class per method.

Each corrector sees the stream batch by batch: scores(z) first adds the batch to its reference statistics (the current
batch is included, like Ramen writing before retrieval) and returns (B, C) scores whose argmax is the prediction.

    c = Csls(W, kappa=20, beta=1.0)
    for s, e in batches: pred = c.scores(Z[s:e]).argmax(1)

ZeroShot            cos_ic
Csls                cos_ic - beta * h_c,        h_c = mean of the kappa largest cos_jc over the reference queries
InvertedSoftmax     tau cos_ic - log sum_j exp(tau cos_jc)   (the class's total soft mass over the reference queries)
Centring            cosine of (z - mean z) and (w - mean w)
PermutedCsls        CONTROL: Csls with its h_c shuffled across classes; same size of correction, wrong classes
No method here uses labels.
"""
import torch


class Corrector:
    name = 'corrector'

    def __init__(self, W):
        self.W = W

    def scores(self, z):
        raise NotImplementedError


class ZeroShot(Corrector):
    name = 'zero-shot'

    def scores(self, z):
        return z @ self.W.T


class Csls(Corrector):
    """Cross-domain similarity local scaling: subtract each class's mean similarity to its kappa nearest queries."""

    def __init__(self, W, kappa=20, beta=1.0):
        super().__init__(W)
        self.kappa, self.beta = kappa, beta
        self.name = f'CSLS k={kappa} b={beta:g}'
        self.top = torch.full((W.shape[0], kappa), -float('inf'), device=W.device)

    def update(self, cos):
        self.top = torch.cat([self.top, cos.T], dim=1).topk(self.kappa, dim=1).values

    def hub(self):
        return self.top.mean(dim=1)  # kappa <= batch size, so after the first batch every entry is finite

    def scores(self, z):
        cos = z @ self.W.T
        self.update(cos)
        return cos - self.beta * self.hub()


class PermutedCsls(Csls):
    def __init__(self, W, kappa=20, beta=1.0, seed=0):
        super().__init__(W, kappa, beta)
        self.name = f'permuted {self.name}'
        g = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(W.shape[0], generator=g).to(W.device)

    def hub(self):
        return super().hub()[self.perm]


class InvertedSoftmax(Corrector):
    """Dynamic inverted softmax (QB-Norm style) with the stream so far as the query bank."""

    def __init__(self, W, tau=50.0):
        super().__init__(W)
        self.tau, self.name = tau, f'inverted softmax t={tau:g}'
        self.log_mass = torch.full((W.shape[0],), -float('inf'), device=W.device)

    def scores(self, z):
        cos = z @ self.W.T
        self.log_mass = torch.logaddexp(self.log_mass, torch.logsumexp(self.tau * cos, dim=0))
        return self.tau * cos - self.log_mass


class Centring(Corrector):
    name = 'centring'

    def __init__(self, W):
        super().__init__(W)
        w = W - W.mean(dim=0, keepdim=True)
        self.Wc = w / w.norm(dim=1, keepdim=True)
        self.sum, self.n = torch.zeros(W.shape[1], device=W.device), 0

    def scores(self, z):
        self.sum, self.n = self.sum + z.sum(dim=0), self.n + len(z)
        zc = z - self.sum / self.n
        return (zc / zc.norm(dim=1, keepdim=True)) @ self.Wc.T


def run(corrector, Z, batches):
    """Predictions of one corrector over the whole stream (numpy argmax per row)."""
    out = torch.empty(len(Z), dtype=torch.long, device=Z.device)
    for s, e in batches:
        out[s:e] = corrector.scores(Z[s:e]).argmax(1)
    return out.cpu().numpy()
