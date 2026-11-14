"""Online triplet mining strategies including Semi-Hard negative selection."""

from typing import Literal

import torch


def compute_pairwise_distances(embeddings: torch.Tensor, squared: bool = True) -> torch.Tensor:
    """Computes the (B, B) pairwise Euclidean distance matrix for L2-normalized embeddings.

    For unit vectors (||u|| = ||v|| = 1), squared Euclidean distance simplifies to:
        ||u - v||^2 = ||u||^2 + ||v||^2 - 2*(u . v) = 2 - 2*(u . v)

    Args:
        embeddings: L2-normalized tensor of shape ``(B, d)``.
        squared: If ``True``, returns squared Euclidean distances ``||u - v||^2``.
            If ``False``, returns Euclidean distances ``||u - v||``.

    Returns:
        Pairwise distance matrix of shape ``(B, B)`` with non-negative entries.

    Raises:
        ValueError: If ``embeddings`` is not a 2-D tensor.

    Notes:
        - **Why avoid ``torch.cdist`` or expansion:** Expanding ``||u - v||^2`` via
          ``2 - 2 * (u @ u.T)`` exploits unit-norm geometry with a single matrix
          multiplication and avoids catastrophic cancellation on near-identical vectors.
        - **Numerical clamping:** Floating-point rounding can cause ``u . u`` to slightly
          exceed ``1.0`` (e.g., ``1.0000001``), making ``2 - 2*(u . u) < 0``. Clamping at
          ``0.0`` prevents ``NaN`` values when taking a square root.
        - **Gradient-safe square root:** The derivative of ``sqrt(x)`` is ``1 / (2 * sqrt(x))``,
          which diverges to ``inf`` on the diagonal where ``x == 0``. Replacing zeros with
          ``1.0`` before ``sqrt`` and masking them back to ``0.0`` via ``torch.where`` keeps
          backward gradients finite across all dtypes (``float16``, ``bfloat16``, ``float32``).
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2-D embeddings of shape (B, d), got {tuple(embeddings.shape)}."
        )

    # Dot product matrix of shape (B, B)
    dot_product = torch.matmul(embeddings, embeddings.t())

    # For unit vectors, ||u||^2 = 1.0, so ||u - v||^2 = 2 - 2 * (u . v)
    distances_squared = torch.clamp(2.0 - 2.0 * dot_product, min=0.0)

    # Zero out self-distances on the diagonal so float32 rounding (~1e-7)
    # does not become ~7e-4 after sqrt()
    diag_mask = torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
    distances_squared = distances_squared.masked_fill(diag_mask, 0.0)

    if not squared:
        is_zero = distances_squared <= 0.0
        safe_sq = torch.where(is_zero, torch.ones_like(distances_squared), distances_squared)
        return torch.where(is_zero, torch.zeros_like(distances_squared), torch.sqrt(safe_sq))

    return distances_squared


def get_anchor_positive_and_negative_masks(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds boolean masks for valid positive and negative pairs in a batch.

    Args:
        labels: 1-D tensor of shape ``(B,)`` containing integer identity class IDs.

    Returns:
        A tuple ``(pos_mask, neg_mask)`` of ``(B, B)`` boolean tensors:
            - ``pos_mask[i, j]`` is ``True`` iff ``labels[i] == labels[j]`` and ``i != j``.
            - ``neg_mask[i, j]`` is ``True`` iff ``labels[i] != labels[j]``.

    Raises:
        ValueError: If ``labels`` is not a 1-D tensor.

    Notes:
        - **Self-pair exclusion:** Every sample has distance ``0`` to itself, so the
          diagonal ``(i == j)`` must be masked out of ``pos_mask``; otherwise every anchor
          would trivially pair with itself as an effortless positive.
    """
    if labels.ndim != 1:
        raise ValueError(f"Expected 1-D labels of shape (B,), got {tuple(labels.shape)}.")

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

    pos_mask = labels_equal & (~indices_equal)
    neg_mask = ~labels_equal

    return pos_mask, neg_mask


def mine_semi_hard_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    fallback: Literal["hardest", "none"] = "hardest",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mines online semi-hard negative triplets from a mini-batch.

    For each valid anchor-positive pair ``(a, p)``, searches for negatives ``n`` lying
    inside the semi-hard margin band:

        D(a, p) < D(a, n) < D(a, p) + margin

    where ``D(., .)`` is the squared Euclidean distance.

    Args:
        embeddings: L2-normalized embeddings of shape ``(B, d)``.
        labels: 1-D identity labels of shape ``(B,)``.
        margin: Margin threshold ``alpha > 0`` matching ``TripletMarginLoss.margin``.
        fallback: Strategy when no negative lies strictly inside the semi-hard band:
            - ``'hardest'``: Selects the closest negative ``argmin_n D(a, n)`` across all
              negatives in the batch so every ``(a, p)`` pair yields a triplet.
            - ``'none'``: Skips the ``(a, p)`` pair (strict FaceNet semi-hard mining).

    Returns:
        A tuple ``(anchors, positives, negatives)``, each of shape ``(M, d)``, where ``M``
        is the number of mined triplets. Returns ``(0, d)`` tensors if no valid triplets
        can be formed.

    Raises:
        ValueError: If ``margin <= 0``, ``fallback`` is invalid, or batch dimensions mismatch.

    Notes:
        - **Three regimes of negatives for an anchor-positive pair ``(a, p)``:**
          1. *Hard negatives* (``D(a, n) <= D(a, p)``): Closer to the anchor than the
             positive is. Produces large gradients, which can destabilize early training
             if driven by mislabeled or heavily occluded faces.
          2. *Semi-hard negatives* (``D(a, p) < D(a, n) < D(a, p) + margin``): Further
             than the positive, yet still violating the margin. Provides steady, informative
             gradients without collapse.
          3. *Easy negatives* (``D(a, n) >= D(a, p) + margin``): Already satisfy the margin
             and produce zero loss/gradient in ``TripletMarginLoss``.
        - **Why PK sampling matters:** A batch needs at least two samples per identity
          (``K >= 2``) and at least two distinct identities (``P >= 2``) so that both
          ``pos_mask`` and ``neg_mask`` are non-empty.
    """
    if margin <= 0:
        raise ValueError(f"margin must be > 0, got {margin}")
    if fallback not in ("hardest", "none"):
        raise ValueError(f"Unsupported fallback: '{fallback}'. Expected 'hardest' or 'none'.")
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.size(0) != labels.size(0):
        raise ValueError(
            f"Expected embeddings (B, d) and labels (B,), got "
            f"embeddings={tuple(embeddings.shape)} and labels={tuple(labels.shape)}."
        )

    batch_size = embeddings.size(0)
    pdist = compute_pairwise_distances(embeddings, squared=True)
    pos_mask, neg_mask = get_anchor_positive_and_negative_masks(labels)

    anchor_list: list[torch.Tensor] = []
    positive_list: list[torch.Tensor] = []
    negative_list: list[torch.Tensor] = []

    for a_idx in range(batch_size):
        pos_indices = torch.where(pos_mask[a_idx])[0]
        neg_indices = torch.where(neg_mask[a_idx])[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        for p_idx in pos_indices:
            d_ap = pdist[a_idx, p_idx]
            d_an = pdist[a_idx, neg_indices]

            # Semi-hard band: D(a, p) < D(a, n) < D(a, p) + margin
            semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + margin)
            semi_hard_candidates = neg_indices[semi_hard_mask]

            if len(semi_hard_candidates) > 0:
                cand_distances = pdist[a_idx, semi_hard_candidates]
                best_neg_idx = semi_hard_candidates[torch.argmin(cand_distances)]
            elif fallback == "hardest":
                best_neg_idx = neg_indices[torch.argmin(d_an)]
            else:
                continue

            anchor_list.append(embeddings[a_idx])
            positive_list.append(embeddings[p_idx])
            negative_list.append(embeddings[best_neg_idx])

    if not anchor_list:
        empty = embeddings[:0]
        return empty, empty, empty

    return (
        torch.stack(anchor_list),
        torch.stack(positive_list),
        torch.stack(negative_list),
    )
