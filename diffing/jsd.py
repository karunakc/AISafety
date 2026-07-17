"""
Jensen-Shannon divergence (and related f-divergences) between the next-token
distributions of two models, computed directly from logits.

Given P = softmax(logits_a), Q = softmax(logits_b), and M = (P+Q)/2:

    JSD(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)     (nats, bounded in [0, ln 2])

Everything is computed in log-space (log_softmax + logsumexp for M) so no
term ever takes log(0): a token a model assigns ~zero probability just gets a
very negative (but finite) log-prob, never -inf/NaN. `eps` is a defensive
floor (in probability units) clamped onto the log-probs, guarding only
against literally-infinite input logits (e.g. an explicitly masked position).

All four metrics below share the same call signature and shape contract:

    metric_fn(logits_a, logits_b, reduction="none", eps=1e-8, mask=None)

logits_a/logits_b: [batch, seq, vocab] or [seq, vocab] (unbatched input is
returned unbatched too). `mask`, if given ([batch, seq] or [seq], 1 = real
token / 0 = padding), excludes padded positions from "mean"/"sequence"
reductions -- ignored for reduction="none".

Reductions:
    "none"      per-token divergence, shape [batch, seq] (or [seq] unbatched)
    "sequence"  averaged over seq_len, shape [batch] (or scalar unbatched)
    "mean"      averaged over every unmasked token in the batch, scalar
"""

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

Reduction = str  # "none" | "sequence" | "mean"


def _log_probs(logits: torch.Tensor, eps: float) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1).clamp_min(math.log(eps))


def _reduce(per_token: torch.Tensor, reduction: Reduction, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if reduction == "none":
        return per_token
    if mask is None:
        if reduction == "sequence":
            return per_token.mean(dim=-1)
        if reduction == "mean":
            return per_token.mean()
    else:
        mask = mask.to(per_token.dtype)
        if reduction == "sequence":
            return (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        if reduction == "mean":
            return (per_token * mask).sum() / mask.sum().clamp_min(1)
    raise ValueError(f"Unknown reduction {reduction!r}, expected 'none', 'sequence', or 'mean'")


def _kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(P||Q) from log-probabilities. Not symmetric: _kl(a, b) != _kl(b, a)."""
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def _jsd(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    log_m = torch.logsumexp(torch.stack([log_p, log_q]), dim=0) - math.log(2)
    return 0.5 * _kl(log_p, log_m) + 0.5 * _kl(log_q, log_m)


def _total_variation(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (log_p.exp() - log_q.exp()).abs().sum(dim=-1)


def _hellinger(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    sqrt_p, sqrt_q = (0.5 * log_p).exp(), (0.5 * log_q).exp()
    return (1.0 / math.sqrt(2)) * torch.sqrt(((sqrt_p - sqrt_q) ** 2).sum(dim=-1))


def _make_metric(divergence_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    """Wraps a (log_p, log_q) -> per_token divergence function with the
    shared squeeze/log-prob/reduction machinery every metric here needs."""

    def metric(logits_a: torch.Tensor, logits_b: torch.Tensor, reduction: Reduction = "none",
               eps: float = 1e-8, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        squeeze = logits_a.dim() == 2
        if squeeze:
            logits_a, logits_b = logits_a.unsqueeze(0), logits_b.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        log_p = _log_probs(logits_a, eps)
        log_q = _log_probs(logits_b, eps)
        per_token = divergence_fn(log_p, log_q)  # [batch, seq]

        result = _reduce(per_token, reduction, mask)
        if squeeze and reduction != "mean":
            result = result.squeeze(0)
        return result

    return metric


jsd_from_logits = _make_metric(_jsd)
kl_from_logits = _make_metric(_kl)
total_variation_from_logits = _make_metric(_total_variation)
hellinger_from_logits = _make_metric(_hellinger)

# Registry so evaluation code (diffing/method8_jsd.py) can request any of
# these by name without branching -- add a new metric by writing a
# (log_p, log_q) -> per_token divergence function and wrapping it with
# _make_metric, then registering it here.
METRICS = {
    "jsd": jsd_from_logits,
    "kl": kl_from_logits,
    "tv": total_variation_from_logits,
    "hellinger": hellinger_from_logits,
}

__all__ = [
    "jsd_from_logits", "kl_from_logits", "total_variation_from_logits", "hellinger_from_logits", "METRICS",
]
