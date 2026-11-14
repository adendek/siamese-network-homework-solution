"""Verification metrics: Accuracy, ROC-AUC, EER, and evaluation pair generation."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve


@dataclass
class VerificationMetrics:
    """Container for 1:1 biometric verification evaluation metrics.

    Attributes:
        best_accuracy: Highest binary classification accuracy across candidate thresholds.
        best_threshold: Squared Euclidean distance threshold achieving ``best_accuracy``.
        eer: Equal Error Rate (operating point where ``FPR == FNR``). Lower is better
            (``0.0`` is ideal; ``0.5`` is random guessing).
        auc: Area Under the ROC Curve (``1.0`` is ideal; ``0.5`` is random guessing).
        mean_pos_dist: Average squared distance across genuine (same-identity) pairs.
        mean_neg_dist: Average squared distance across impostor (different-identity) pairs.
    """

    best_accuracy: float
    best_threshold: float
    eer: float
    auc: float
    mean_pos_dist: float
    mean_neg_dist: float


def generate_evaluation_pairs(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_pairs: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates balanced same-identity and different-identity evaluation pairs.

    Args:
        embeddings: Embedding tensor of shape ``(N, d)``.
        labels: 1-D tensor of shape ``(N,)`` containing integer identity IDs.
        num_pairs: Total number of pairs to draw (split evenly into ``num_pairs // 2``
            genuine positive pairs and ``num_pairs // 2`` impostor negative pairs).
            Must be an even integer ``>= 2``.
        seed: Random seed for deterministic pair sampling across validation epochs.

    Returns:
        A tuple ``(distances, actual_matches)``:
            - ``distances``: 1-D ``float64`` array of shape ``(num_pairs,)`` containing
              pairwise squared Euclidean distances ``||z_i - z_j||^2``.
            - ``actual_matches``: 1-D ``int64`` array of shape ``(num_pairs,)`` with
              ``1`` for same-identity pairs and ``0`` for different-identity pairs.

    Raises:
        ValueError: If input shapes are invalid, ``num_pairs < 2`` or odd, no identity
            has at least 2 samples, or fewer than 2 distinct identities exist.

    Notes:
        - **Why deterministic pair sampling matters during validation:** Validation datasets
          with ``N`` images have ``N * (N - 1) / 2`` possible pairs, heavily skewed toward
          impostors. Drawing a balanced 50/50 subset with a fixed ``seed`` ensures that
          epoch-to-epoch changes in EER and AUC reflect genuine model improvement rather
          than random pair resampling noise.
        - **Why squared Euclidean distance:** For L2-normalized embeddings,
          ``||z_i - z_j||^2`` lies in ``[0.0, 4.0]`` and matches the distance metric used
          in ``mine_semi_hard_triplets`` and ``TripletMarginLoss``.
    """
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.size(0) != labels.size(0):
        raise ValueError(
            f"Expected embeddings (N, d) and labels (N,), got "
            f"embeddings={tuple(embeddings.shape)} and labels={tuple(labels.shape)}."
        )
    if num_pairs < 2 or num_pairs % 2 != 0:
        raise ValueError(f"num_pairs must be an even integer >= 2, got {num_pairs}.")

    rng = np.random.default_rng(seed)

    emb_np = embeddings.detach().cpu().numpy()
    lbl_np = labels.detach().cpu().numpy()

    unique_labels = np.unique(lbl_np)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Need at least 2 distinct identities to form negative pairs, "
            f"got {len(unique_labels)}."
        )

    label_to_indices = {lbl: np.where(lbl_np == lbl)[0] for lbl in unique_labels}
    labels_with_pairs = [lbl for lbl, idxs in label_to_indices.items() if len(idxs) >= 2]
    if not labels_with_pairs:
        raise ValueError("Need at least one identity with >= 2 samples to form positive pairs.")

    half_pairs = num_pairs // 2
    distances: list[float] = []
    targets: list[int] = []

    # 1. Generate Positive Pairs (Same Identity, distinct indices)
    for _ in range(half_pairs):
        lbl = rng.choice(labels_with_pairs)
        idx1, idx2 = rng.choice(label_to_indices[lbl], size=2, replace=False)
        dist = float(np.sum((emb_np[idx1] - emb_np[idx2]) ** 2))
        distances.append(dist)
        targets.append(1)

    # 2. Generate Negative Pairs (Different Identities)
    for _ in range(half_pairs):
        lbl1, lbl2 = rng.choice(unique_labels, size=2, replace=False)
        idx1 = rng.choice(label_to_indices[lbl1])
        idx2 = rng.choice(label_to_indices[lbl2])
        dist = float(np.sum((emb_np[idx1] - emb_np[idx2]) ** 2))
        distances.append(dist)
        targets.append(0)

    return np.asarray(distances, dtype=np.float64), np.asarray(targets, dtype=np.int64)


def evaluate_verification(
    distances: np.ndarray, actual_matches: np.ndarray
) -> VerificationMetrics:
    """Computes biometric verification accuracy, ROC-AUC, and Equal Error Rate (EER).

    Args:
        distances: 1-D array of pairwise distances (smaller = more similar).
        actual_matches: 1-D binary ground-truth array (``1`` = genuine match,
            ``0`` = impostor non-match).

    Returns:
        Populated ``VerificationMetrics`` dataclass.

    Raises:
        ValueError: If inputs are empty, have mismatched shapes, or contain only one class.

    Notes:
        - **Sign inversion for ROC curves:** ``sklearn.metrics.roc_curve`` expects
          *similarity scores* where higher values predict the positive class (``1``).
          Because smaller distances indicate stronger matches, we pass
          ``scores = -distances``.
        - **Equal Error Rate (EER):** In biometric verification, False Accept Rate
          (``FAR == FPR``) measures impostors wrongly accepted, while False Reject Rate
          (``FRR == FNR = 1 - TPR``) measures authorized users wrongly rejected. EER is
          the point on the ROC curve where ``FPR == FNR``.
        - **Exact threshold search:** Evaluating candidate thresholds at the finite
          decision boundaries produced by ``roc_curve`` (plus a uniform grid fallback)
          finds the exact threshold maximizing binary accuracy ``(distances <= tau)``.
    """
    distances = np.asarray(distances, dtype=np.float64)
    actual_matches = np.asarray(actual_matches, dtype=np.int64)

    if distances.ndim != 1 or distances.shape != actual_matches.shape or distances.size == 0:
        raise ValueError(
            f"Expected non-empty 1-D arrays of matching shape, got "
            f"distances={distances.shape} and actual_matches={actual_matches.shape}."
        )

    pos_mask = actual_matches == 1
    neg_mask = actual_matches == 0
    if not np.any(pos_mask) or not np.any(neg_mask):
        raise ValueError(
            "actual_matches must contain both positive (1) and negative (0) pairs."
        )

    # Smaller distance -> higher similarity score
    scores = -distances
    fpr, tpr, roc_thresholds = roc_curve(actual_matches, scores)
    roc_auc = float(auc(fpr, tpr))
    fnr = 1.0 - tpr

    # Equal Error Rate (where FPR ~= FNR)
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    # Candidate distance thresholds: combine uniform grid with finite ROC change points
    d_min = min(0.0, float(distances.min()))
    d_max = max(4.0, float(distances.max()))
    finite_roc_taus = -roc_thresholds[np.isfinite(roc_thresholds)]
    cand_thresholds = np.unique(
        np.concatenate([np.linspace(d_min, d_max, num=400), finite_roc_taus])
    )

    best_acc = -1.0
    best_thresh = float(cand_thresholds[0])

    for tau in cand_thresholds:
        preds = distances <= tau
        acc = float(np.mean(preds == pos_mask))
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(tau)

    return VerificationMetrics(
        best_accuracy=best_acc,
        best_threshold=best_thresh,
        eer=eer,
        auc=roc_auc,
        mean_pos_dist=float(np.mean(distances[pos_mask])),
        mean_neg_dist=float(np.mean(distances[neg_mask])),
    )
