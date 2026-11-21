"""
Per-image alignment metrics between two gradient directions (pure torch, no model, no state).

Every function takes (B, D) tensors whose row b belongs to image b and returns a (B,) float32 tensor.

    from alignment import cosine, sign_agreement
    cosine(ramen_update, ce_grad)             # in [-1, 1]; > 0 = moving along -update lowers the CE loss
    sign_agreement(ramen_update, ce_grad)     # share of |ce_grad| mass whose sign the SignSGD step gets right
"""
import torch


def cosine(a, b, eps=1e-12):
    """Row-wise cosine similarity, computed in float32."""
    a, b = a.float(), b.float()
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(eps)


def projection_share(part, whole, ref, eps=1e-12):
    """<part, ref> / (||whole|| ||ref||): the part's additive share of cosine(whole, ref) when whole = sum of parts."""
    part, whole, ref = part.float(), whole.float(), ref.float()
    return (part * ref).sum(1) / (whole.norm(dim=1) * ref.norm(dim=1)).clamp_min(eps)


def sign_agreement(step, ref, eps=1e-12):
    """
    sum_t |ref_t| 1[sign(step_t) = sign(ref_t)] / sum_t |ref_t|.

    The share of the reference gradient's mass on coordinates where a SignSGD step along `step` moves the same way as
    a step along `ref`. 0.5 = no better than random signs; 1.0 = every coordinate right.
    """
    step, ref = step.float(), ref.float()
    w = ref.abs()
    return (w * (torch.sign(step) == torch.sign(ref)).float()).sum(1) / w.sum(1).clamp_min(eps)
