"""Geometric diagnostics for contrastive embeddings on the unit hypersphere.

These metrics support PhD-style analysis of representation quality. All metrics
are computed *per training step* on the current mini-batch and are intended for
logging only — they have no effect on the loss or gradients.

References
----------
* Wang & Isola, 2020, "Understanding Contrastive Representation Learning through
  Alignment and Uniformity on the Hypersphere" — ``arXiv:2005.10242``.
* Roy & Vetterli, 2007, "The effective rank: A measure of effective
  dimensionality."
* Ethayarajh, 2019, "How Contextual are Contextualized Word Representations?
  Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings."

The block of metrics returned per call covers:

- **Tau**: current value of the (possibly learnable) temperature.
- **Alignment**: $E_{(q, p^+)}\|q - p^+\|_2^2$ on the unit sphere, equal to
  $2 - 2 E[\langle q, p^+\rangle]$.
- **UniformityQ**, **UniformityD**: $\log E_{x, y}\,e^{-2 \|x - y\|_2^2}$ over
  pairs within queries and within documents respectively.
- **PositiveSimMean / Std**: cosine similarity of the matched diagonal pairs.
- **NegativeSimMean / Std / Max**: cosine similarity over off-diagonal pairs.
- **SimGap**: positive mean minus negative mean.
- **EffectiveRankQ / EffectiveRankD**: $\exp(H(\sigma))$ for normalized
  singular values $\sigma$ — a soft proxy for hypersphere coverage dimension.
- **AnisotropyQ / AnisotropyD**: top-1 singular value squared over the
  Frobenius norm squared (alias of Ethayarajh anisotropy).
- **EmbedNormMeanQ / EmbedNormStdQ**: should be close to ``1.0`` and ``0.0``
  for L2-normalized representations; deviations flag a bug or collapsed view.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def _safe_norm_stats(x: torch.Tensor) -> tuple[float, float]:
    norms = x.norm(p=2, dim=-1)
    return float(norms.mean().item()), float(norms.std(unbiased=False).item())


@torch.no_grad()
def _uniformity(x: torch.Tensor) -> float:
    """Wang & Isola 2020 uniformity: log E_{x,y}[exp(-2 ||x - y||^2)].

    Computed via expanded squared distance on (already-normalized) embeddings.
    Returns ``nan`` if the batch has fewer than 2 rows.
    """
    n = x.shape[0]
    if n < 2:
        return float("nan")
    sq_dist = torch.cdist(x, x, p=2).pow(2)  # [n, n]
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return float(torch.log(torch.exp(-2.0 * sq_dist[mask]).mean()).item())


@torch.no_grad()
def _effective_rank(x: torch.Tensor) -> tuple[float, float]:
    """Effective rank exp(H(s)) and anisotropy index s_max^2 / sum(s^2).

    Computed from singular values of the centered embedding matrix, capped at
    min(B, D) singular values. Falls back to ``(nan, nan)`` if SVD fails or
    batch has fewer than 2 rows.
    """
    if x.shape[0] < 2:
        return float("nan"), float("nan")
    centered = x - x.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered.float())
    except Exception:
        return float("nan"), float("nan")
    s = s[s > 1e-12]
    if s.numel() == 0:
        return float("nan"), float("nan")
    s_sq = s.pow(2)
    p = s / s.sum()
    entropy = -(p * (p + 1e-12).log()).sum()
    eff_rank = float(math.exp(float(entropy.item())))
    anisotropy = float((s_sq.max() / s_sq.sum()).item())
    return eff_rank, anisotropy


@torch.no_grad()
def embedding_geometry_metrics(
    q_vecs: torch.Tensor,
    d_vecs: torch.Tensor,
    *,
    tau: float | torch.Tensor | None = None,
    prefix: str = "Geom/",
) -> dict[str, float]:
    """Compute a panel of N-sphere geometric metrics for a contrastive batch.

    Args:
        q_vecs: Query embeddings, expected L2-normalized, shape ``[B, D]``.
        d_vecs: Positive document embeddings, shape ``[B, D]``.
        tau: Current temperature value (for logging).
        prefix: Key prefix for the returned dict so multiple panels can coexist.

    Returns:
        Flat dictionary of float metrics keyed by ``prefix + Name``.
    """
    metrics: dict[str, float] = {}
    batch_size = int(q_vecs.shape[0])
    metrics[f"{prefix}BatchSize"] = float(batch_size)

    if tau is not None:
        tau_f = float(tau.item()) if isinstance(tau, torch.Tensor) else float(tau)
        metrics[f"{prefix}Tau"] = tau_f
        metrics[f"{prefix}InvTau"] = float(1.0 / tau_f) if tau_f > 0 else float("inf")
        metrics[f"{prefix}LogTau"] = float(math.log(tau_f)) if tau_f > 0 else float("nan")

    norm_mean_q, norm_std_q = _safe_norm_stats(q_vecs)
    norm_mean_d, norm_std_d = _safe_norm_stats(d_vecs)
    metrics[f"{prefix}EmbedNormMeanQ"] = norm_mean_q
    metrics[f"{prefix}EmbedNormStdQ"] = norm_std_q
    metrics[f"{prefix}EmbedNormMeanD"] = norm_mean_d
    metrics[f"{prefix}EmbedNormStdD"] = norm_std_d

    sim = q_vecs @ d_vecs.t()  # raw cosine if both unit-norm
    pos_sim = sim.diagonal()
    metrics[f"{prefix}PositiveSimMean"] = float(pos_sim.mean().item())
    metrics[f"{prefix}PositiveSimStd"] = float(pos_sim.std(unbiased=False).item())

    if batch_size > 1:
        eye = torch.eye(batch_size, dtype=torch.bool, device=sim.device)
        neg_sim = sim[~eye]
        metrics[f"{prefix}NegativeSimMean"] = float(neg_sim.mean().item())
        metrics[f"{prefix}NegativeSimStd"] = float(neg_sim.std(unbiased=False).item())
        metrics[f"{prefix}NegativeSimMax"] = float(neg_sim.max().item())
        metrics[f"{prefix}SimGap"] = metrics[f"{prefix}PositiveSimMean"] - metrics[f"{prefix}NegativeSimMean"]
    else:
        metrics[f"{prefix}NegativeSimMean"] = float("nan")
        metrics[f"{prefix}NegativeSimStd"] = float("nan")
        metrics[f"{prefix}NegativeSimMax"] = float("nan")
        metrics[f"{prefix}SimGap"] = float("nan")

    metrics[f"{prefix}Alignment"] = 2.0 - 2.0 * metrics[f"{prefix}PositiveSimMean"]
    metrics[f"{prefix}UniformityQ"] = _uniformity(q_vecs)
    metrics[f"{prefix}UniformityD"] = _uniformity(d_vecs)

    eff_rank_q, aniso_q = _effective_rank(q_vecs)
    eff_rank_d, aniso_d = _effective_rank(d_vecs)
    metrics[f"{prefix}EffectiveRankQ"] = eff_rank_q
    metrics[f"{prefix}EffectiveRankD"] = eff_rank_d
    metrics[f"{prefix}AnisotropyQ"] = aniso_q
    metrics[f"{prefix}AnisotropyD"] = aniso_d

    return metrics
