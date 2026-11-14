"""Vectorized Triplet Margin Loss implementation."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss(nn.Module):
    """Vectorized Triplet Margin Loss from scratch.

    Computes the triplet objective over batches of pre-mined (anchor, positive, negative)
    embedding tuples:

        L(a, p, n) = max(0, ||z_a - z_p||^2 - ||z_a - z_n||^2 + margin)

    Args:
        margin: Separation margin alpha > 0 enforcing a minimum distance gap between
            positive and negative pairs. Defaults to 0.2.
        reduction: Specifies how per-triplet losses are aggregated across the batch:
            - ``'mean'``: Average loss over all M triplets (returns 0.0 if M == 0).
            - ``'sum'``: Sum of losses over all M triplets.
            - ``'none'``: Unreduced 1-D tensor of shape ``(M,)``.

    Raises:
        ValueError: If ``margin <= 0`` or ``reduction`` is not one of
            ``('mean', 'sum', 'none')``.

    Notes:
        - **Geometric intuition:** The loss encourages the anchor ``z_a`` to be closer to
          the same-identity positive ``z_p`` than to the different-identity negative ``z_n``
          by at least ``margin``: ``d(a, n) >= d(a, p) + margin``.
        - **Why squared Euclidean distance:** Using ``||u - v||^2`` instead of ``||u - v||``
          avoids a ``sqrt`` operation (which has an undefined gradient at zero when
          ``u == v``). Moreover, when embeddings are L2-normalized onto the unit
          hypersphere (``||u|| = ||v|| = 1``), squared Euclidean distance simplifies directly
          to cosine distance: ``||u - v||^2 = 2 - 2 * (u · v)``.
        - **Active vs. easy triplets:** ``F.relu(...)`` clamps negative values to ``0.0``.
          Any triplet that already satisfies ``d(a, n) >= d(a, p) + margin`` produces zero
          loss and zero gradient ("easy triplet"), allowing optimization to focus solely
          on margin-violating ("active") triplets.
        - **Gradient direction:** For an active triplet, the analytical gradients are:
          ``dL/dz_a = 2 * (z_n - z_p)``, ``dL/dz_p = 2 * (z_p - z_a)``, and
          ``dL/dz_n = 2 * (z_a - z_n)``, simultaneously pulling ``(a, p)`` together and
          pushing ``(a, n)`` apart.
        - **Alignment with online mining:** Ensure that both the ``margin`` value and the
          use of *squared* Euclidean distance match the distance definition used in your
          triplet miner (e.g., ``mine_semi_hard_triplets``).

    Example:
        >>> criterion = TripletMarginLoss(margin=0.2, reduction="mean")
        >>> a = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
        >>> p = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
        >>> n = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
        >>> loss = criterion(a, p, n)
    """

    VALID_REDUCTIONS = ("mean", "sum", "none")

    def __init__(
        self,
        margin: float = 0.2,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initializes the TripletMarginLoss."""
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be > 0, got {margin}")
        if reduction not in self.VALID_REDUCTIONS:
            raise ValueError(
                f"Unsupported reduction: '{reduction}'. "
                f"Expected one of {self.VALID_REDUCTIONS}."
            )

        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Computes triplet margin loss over a batch of mined embeddings.

        Args:
            anchors: Anchor embeddings of shape ``(M, d)``.
            positives: Positive embeddings (same identity as anchor) of shape ``(M, d)``.
            negatives: Negative embeddings (different identity) of shape ``(M, d)``.

        Returns:
            Loss tensor: a 0-D scalar when ``reduction`` is ``'mean'`` or ``'sum'``, or a
            1-D tensor of shape ``(M,)`` when ``reduction='none'``.

        Raises:
            ValueError: If inputs are not 2-D tensors or do not share identical ``(M, d)``
                shapes.

        Notes:
            - **Vectorization:** Computing ``((anchors - positives) ** 2).sum(dim=1)``
              evaluates all ``M`` triplets in parallel without Python loops.
            - **Empty batch safety (``M == 0``):** If a miner finds zero valid triplets in a
              batch, ``losses`` has shape ``(0,)``. Calling ``losses.mean()`` on an empty
              tensor would divide ``0 / 0`` and produce ``NaN``. Falling back to
              ``losses.sum()`` returns ``0.0`` while keeping the tensor connected to the
              PyTorch autograd graph so ``loss.backward()`` executes safely.
        """
        if anchors.ndim != 2 or not (anchors.shape == positives.shape == negatives.shape):
            raise ValueError(
                f"Expected 2-D tensors with identical shapes (M, d), got "
                f"anchors={tuple(anchors.shape)}, positives={tuple(positives.shape)}, "
                f"negatives={tuple(negatives.shape)}."
            )

        # Squared Euclidean distance: ||u - v||^2 = sum((u - v)^2, dim=1)
        d_ap = torch.sum((anchors - positives) ** 2, dim=1)
        d_an = torch.sum((anchors - negatives) ** 2, dim=1)

        # Margin penalty: clamp negative values (easy triplets) to 0
        losses = F.relu(d_ap - d_an + self.margin)

        if self.reduction == "mean":
            return losses.sum() if losses.numel() == 0 else losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses
