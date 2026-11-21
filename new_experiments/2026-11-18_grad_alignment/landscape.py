"""
Geometry of the loss-landscape slices (pure torch / numpy; no model, no state).

Parameter space (evaluated on the GPU by RamenLandscapeProbe, landscape_methods.py):
    u = theta_i - theta_0                Ramen's actual step (read from the parameters after SignSGD)
    o = -lr * sign(g_ce)                 ORACLE: the SignSGD step on the true-label cross-entropy
    e1 = u,  e2 = (o - <o,u>/<u,u> u) rescaled to ||u||        (orthonormal up to the common scale ||u||)
    theta(a, b) = theta_0 + a e1 + b e2  -> theta_0 at (0, 0), Ramen's theta_i at (1, 0), oracle step at oracle_xy
Lines from theta_0 along u, o and a random sign step r = -lr * sign(N(0, I)) (same length: lr sqrt(D)).

Embedding space (offline, from z0, z1 and the frozen text head W):
    z(a, b) = normalise(z0 + a t1 + b t2), t1 = z1 - z0, t2 = the true-class ascent direction of the head at z0
    (w_y - sum_c p_c w_c), made orthogonal to z0 and t1 and rescaled to ||t1||   -> z0 at (0, 0), z1 at (1, 0)

    from landscape import param_plane, grid_points, embedding_landscape
"""
import numpy as np
import torch

# grid of the 2D parameter slice and of the 1D lines (multiples of ||u|| = lr sqrt(D))
GRID_A = np.round(np.arange(-1.0, 2.5 + 1e-9, 0.125), 4)     # along Ramen's step (29 points)
GRID_B = np.round(np.arange(-1.0, 1.75 + 1e-9, 0.125), 4)    # along the oracle step, orthogonalised (23 points)
LINE_T = np.round(np.arange(-1.0, 3.0 + 1e-9, 0.1), 4)        # 1D lines (41 points)


def param_plane(u, o):
    """(e1, e2, oracle_xy): plane basis for one image (u, o: (D,) float32) and the oracle step's coordinates."""
    u, o = u.float(), o.float()
    uu = (u @ u).clamp_min(1e-12)
    along = (o @ u) / uu
    perp = o - along * u
    scale = perp.norm().clamp_min(1e-12)
    e2 = perp * (u.norm() / scale)
    return u, e2, (float(along), float(scale / u.norm()))


def grid_points(a=GRID_A, b=GRID_B):
    """(n, 2) float32 array of (a, b) for every grid point, row-major over b then a; reshape with (len(b), len(a))."""
    A, B = np.meshgrid(a, b)
    return np.stack([A.ravel(), B.ravel()], 1).astype(np.float32)


def random_sign_step(D, lr, seed, device):
    g = torch.Generator(device='cpu').manual_seed(int(seed))      # own generator: the global RNG is untouched
    return (-lr * torch.sign(torch.randn(D, generator=g))).to(device)


def head_probs(z, W, scale=100.0):
    """CLIP head on normalised features: softmax(scale z W^T). z (..., d), W (C, d)."""
    return torch.softmax(scale * z @ W.T, dim=-1)


def embedding_landscape(z0, z1, W, y, a=GRID_A, b=GRID_B, scale=100.0):
    """
    CE toward y, entropy and the predicted class of the frozen head on the plane through z0 and z1 (see module doc).
    Returns dict of (len(b), len(a)) arrays plus the plane coordinates of the true-class ascent direction.
    """
    z0, z1, W = (torch.as_tensor(v, dtype=torch.float32) for v in (z0, z1, W))
    t1 = z1 - z0
    p0 = head_probs(z0, W, scale)
    asc = W[y] - p0 @ W                                       # -d CE / dz at z0, up to the factor `scale`
    basis = [z0 / z0.norm()]
    t1_tan = t1 - (t1 @ basis[0]) * basis[0]
    if t1_tan.norm() > 1e-12:
        basis.append(t1_tan / t1_tan.norm())
    t2 = asc.clone()
    for e in basis:
        t2 = t2 - (t2 @ e) * e
    t2 = t2 * (t1.norm() / t2.norm().clamp_min(1e-12))
    P = torch.as_tensor(grid_points(a, b))
    z = z0 + P[:, :1] * t1 + P[:, 1:] * t2
    z = z / z.norm(dim=1, keepdim=True)
    p = head_probs(z, W, scale)
    shape = (len(b), len(a))
    return dict(ce=(-torch.log(p[:, y].clamp_min(1e-12))).reshape(shape).numpy(),
                entropy=(-(p * torch.log(p.clamp_min(1e-12))).sum(1)).reshape(shape).numpy(),
                pred=p.argmax(1).reshape(shape).numpy(),
                step_norm=float(t1.norm()))
