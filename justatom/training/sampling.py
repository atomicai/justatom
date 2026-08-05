from __future__ import annotations

import math
import string
from collections import Counter
from collections.abc import Callable

import torch


def sample_negative_derangement(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 to construct negative pairs")

    permutation = torch.arange(batch_size, device=device)
    for idx in range(batch_size - 1, 0, -1):
        swap_idx = int(torch.randint(0, idx, (1,), device=device).item())
        tmp = permutation[idx].clone()
        permutation[idx] = permutation[swap_idx]
        permutation[swap_idx] = tmp
    return permutation


def sample_safe_negative_indices(
    *,
    doc_key_ids: torch.Tensor,
    content_key_ids: torch.Tensor | None = None,
    query_key_ids: torch.Tensor | None = None,
    queries: list[str] | None = None,
    docs: list[str] | None = None,
    lexical_text_by_content: dict[str, str] | None = None,
    inverse_idf_recall_fn: Callable[[str, list[str] | str], float] | None = None,
    max_negative_inverse_idf_recall: float | None = None,
    min_negative_inverse_idf_recall: float | None = None,
    negative_sampling_mode: str = "safe-random",
    hard_negative_top_k: int | None = None,
) -> tuple[torch.Tensor, int]:
    batch_size = int(doc_key_ids.shape[0])
    device = doc_key_ids.device
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 to construct negative pairs")

    negative_indices = torch.empty(batch_size, dtype=torch.long, device=device)
    fallback_count = 0
    all_indices = torch.arange(batch_size, device=device)
    allowed_modes = {"safe-random", "semi-hard-idf", "hard-idf"}
    if negative_sampling_mode not in allowed_modes:
        raise ValueError(f"negative_sampling_mode must be one of: {', '.join(sorted(allowed_modes))}")
    if hard_negative_top_k is not None and hard_negative_top_k < 1:
        raise ValueError("hard_negative_top_k must be >= 1 when provided")

    needs_idf = (
        max_negative_inverse_idf_recall is not None
        or min_negative_inverse_idf_recall is not None
        or negative_sampling_mode != "safe-random"
    )
    if not needs_idf:
        valid = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device).fill_diagonal_(False)
        valid &= doc_key_ids.unsqueeze(0) != doc_key_ids.unsqueeze(1)
        if content_key_ids is not None:
            valid &= content_key_ids.unsqueeze(0) != content_key_ids.unsqueeze(1)
        if query_key_ids is not None:
            valid &= query_key_ids.unsqueeze(0) != query_key_ids.unsqueeze(1)

        any_valid = valid.any(dim=1)
        if not bool(any_valid.all().item()):
            fallback_valid = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device).fill_diagonal_(False)
            fallback_valid &= doc_key_ids.unsqueeze(0) != doc_key_ids.unsqueeze(1)
            if content_key_ids is not None:
                fallback_valid &= content_key_ids.unsqueeze(0) != content_key_ids.unsqueeze(1)
            valid = torch.where((~any_valid).unsqueeze(1), fallback_valid, valid)
            any_valid = valid.any(dim=1)
            if not bool(any_valid.all().item()):
                hard_fallback = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device).fill_diagonal_(False)
                fallback_count = int((~any_valid).sum().item())
                valid = torch.where(any_valid.unsqueeze(1), valid, hard_fallback)

        uniform = torch.rand(batch_size, batch_size, device=device).clamp(1e-9, 1.0 - 1e-9)
        gumbel = -torch.log(-torch.log(uniform))
        return gumbel.masked_fill(~valid, float("-inf")).argmax(dim=1), fallback_count

    for idx in range(batch_size):
        valid_mask = all_indices != idx
        valid_mask &= doc_key_ids != doc_key_ids[idx]
        if content_key_ids is not None:
            valid_mask &= content_key_ids != content_key_ids[idx]
        if query_key_ids is not None:
            valid_mask &= query_key_ids != query_key_ids[idx]
        candidate_indices = all_indices[valid_mask]

        if candidate_indices.numel() == 0:
            fallback_mask = all_indices != idx
            fallback_mask &= doc_key_ids != doc_key_ids[idx]
            if content_key_ids is not None:
                fallback_mask &= content_key_ids != content_key_ids[idx]
            candidate_indices = all_indices[fallback_mask]
        if candidate_indices.numel() == 0:
            fallback_count += 1
            candidate_indices = all_indices[all_indices != idx]

        if queries is not None and docs is not None and inverse_idf_recall_fn is not None:
            query = queries[idx]
            scored: list[tuple[int, float]] = []
            for candidate_idx in candidate_indices.detach().cpu().tolist():
                candidate_doc = docs[candidate_idx]
                sparse_text = (
                    lexical_text_by_content.get(candidate_doc, candidate_doc) if lexical_text_by_content else candidate_doc
                )
                scored.append((candidate_idx, inverse_idf_recall_fn(query, sparse_text)))

            safe = [
                item for item in scored if max_negative_inverse_idf_recall is None or item[1] <= max_negative_inverse_idf_recall
            ]
            preferred = [
                item for item in safe if min_negative_inverse_idf_recall is None or item[1] >= min_negative_inverse_idf_recall
            ]
            if negative_sampling_mode == "safe-random":
                chosen = safe
            else:
                chosen = sorted(preferred or safe, key=lambda item: item[1], reverse=True)
                if hard_negative_top_k is not None:
                    chosen = chosen[:hard_negative_top_k]
            if chosen:
                candidate_indices = torch.tensor(
                    [candidate_idx for candidate_idx, _ in chosen],
                    dtype=torch.long,
                    device=device,
                )

        sampled_offset = int(torch.randint(0, int(candidate_indices.numel()), (1,), device=device).item())
        negative_indices[idx] = candidate_indices[sampled_offset]

    return negative_indices, fallback_count


def inverse_idf_recall(
    query: str,
    doc_text: list[str] | str,
    *,
    stopsyms: str | None = None,
) -> float:
    punctuation = (stopsyms or "«»:\"'") + string.punctuation
    if isinstance(doc_text, list):
        words = [
            word for text in doc_text for word in "".join(char for char in text.lower().strip() if char not in punctuation).split()
        ]
        document_words = Counter(words)
    else:
        document_words = Counter(
            "".join(char for char in word.lower().strip() if char not in punctuation) for word in doc_text.split()
        )

    query_words = "".join(char for char in query if char not in punctuation).lower().strip().split()
    numerator = sum(1.0 / math.log(1 + document_words.get(word, 1)) for word in query_words if word in document_words)
    denominator = sum(1.0 / math.log(1 + document_words.get(word, 1)) for word in query_words)
    return numerator / denominator if denominator > 0 else 0.0
