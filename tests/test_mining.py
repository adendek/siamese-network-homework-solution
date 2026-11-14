"""Unit tests for online triplet mining utilities."""

import math

import pytest
import torch
import torch.nn.functional as F

from src.mining import (
    compute_pairwise_distances,
    get_anchor_positive_and_negative_masks,
    mine_semi_hard_triplets,
)


def _unit_vec(angle_rad: float) -> list[float]:
    """Helper returning a 2-D unit vector at the given angle."""
    return [math.cos(angle_rad), math.sin(angle_rad)]


class TestComputePairwiseDistances:
    def test_known_unit_vectors(self) -> None:
        # Identical (0 deg), orthogonal (90 deg), and opposite (180 deg) unit vectors
        emb = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        d_sq = compute_pairwise_distances(emb, squared=True)
        d_euclid = compute_pairwise_distances(emb, squared=False)

        expected_sq = torch.tensor(
            [
                [0.0, 0.0, 2.0, 4.0],
                [0.0, 0.0, 2.0, 4.0],
                [2.0, 2.0, 0.0, 2.0],
                [4.0, 4.0, 2.0, 0.0],
            ]
        )
        assert d_sq.shape == (4, 4)
        assert torch.allclose(d_sq, expected_sq, atol=1e-6)
        assert torch.allclose(d_euclid, torch.sqrt(expected_sq), atol=1e-6)

    @pytest.mark.parametrize("squared", [True, False])
    def test_matches_cdist_on_normalized_embeddings(self, squared: bool) -> None:
        torch.manual_seed(42)
        emb = F.normalize(torch.randn(16, 64), p=2, dim=1)

        actual = compute_pairwise_distances(emb, squared=squared)
        expected = torch.cdist(emb, emb, p=2)
        if squared:
            expected = expected**2

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_non_squared_gradient_is_finite_at_zero_distance(self) -> None:
        # Duplicate rows produce 0.0 off-diagonal distance; backward must not yield NaN/Inf
        emb = F.normalize(torch.tensor([[1.0, 2.0], [1.0, 2.0], [-1.0, 0.5]]), p=2, dim=1)
        emb.requires_grad_(True)

        dists = compute_pairwise_distances(emb, squared=False)
        dists.sum().backward()

        assert emb.grad is not None
        assert torch.isfinite(emb.grad).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
    @pytest.mark.parametrize("squared", [True, False])
    def test_dtype_preservation(self, dtype: torch.dtype, squared: bool) -> None:
        emb = F.normalize(torch.randn(6, 16, dtype=dtype), p=2, dim=1)
        dists = compute_pairwise_distances(emb, squared=squared)
        assert dists.dtype == dtype

    def test_non_2d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D embeddings"):
            compute_pairwise_distances(torch.randn(16))
        with pytest.raises(ValueError, match="2-D embeddings"):
            compute_pairwise_distances(torch.randn(2, 4, 16))


class TestAnchorPositiveNegativeMasks:
    def test_mask_properties_and_exact_entries(self) -> None:
        labels = torch.tensor([0, 0, 1, 2, 1])
        pos_mask, neg_mask = get_anchor_positive_and_negative_masks(labels)

        assert pos_mask.shape == (5, 5)
        assert neg_mask.shape == (5, 5)
        assert pos_mask.dtype == torch.bool
        assert neg_mask.dtype == torch.bool

        # Diagonal must never be positive or negative
        diag = torch.eye(5, dtype=torch.bool)
        assert not (pos_mask & diag).any()
        assert not (neg_mask & diag).any()

        # Every off-diagonal pair is either positive or negative, never both
        assert not (pos_mask & neg_mask).any()
        assert (pos_mask | neg_mask | diag).all()

        # Verify specific pairs
        assert pos_mask[0, 1] and pos_mask[1, 0]
        assert pos_mask[2, 4] and pos_mask[4, 2]
        assert not pos_mask[3].any()  # label 2 appears only once
        assert neg_mask[0, 2] and neg_mask[0, 3]

    def test_non_1d_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D labels"):
            get_anchor_positive_and_negative_masks(torch.tensor([[0], [1], [0]]))


class TestMineSemiHardTriplets:
    def test_selects_semi_hard_negative_over_hard_and_easy(self) -> None:
        # Place vectors on the 2D unit circle where d^2(0, theta) = 2 - 2*cos(theta):
        # - Anchor a0 at 0 deg: [1, 0]
        # - Positive p1 at 60 deg: d^2(a0, p1) = 2 - 2*cos(60) = 1.0
        # With margin = 0.5, semi-hard band for (a0, p1) is (1.0, 1.5).
        # - n_hard at 45 deg: d^2 = 2 - sqrt(2) ~= 0.586 (hard: < 1.0)
        # - n_semi at 72 deg: d^2 = 2 - 2*cos(72) ~= 1.382 (semi-hard: in (1.0, 1.5))
        # - n_easy at 180 deg: d^2 = 4.0 (easy: > 1.5)
        # Unique labels [1, 2, 3] for negatives ensure only class 0 forms (a, p) pairs.
        a0 = _unit_vec(0.0)
        p1 = _unit_vec(math.radians(60.0))
        n_hard = _unit_vec(math.radians(45.0))
        n_semi = _unit_vec(math.radians(72.0))
        n_easy = _unit_vec(math.radians(180.0))

        embeddings = torch.tensor([a0, p1, n_hard, n_semi, n_easy], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 2, 3])

        anchors, positives, negatives = mine_semi_hard_triplets(
            embeddings, labels, margin=0.5, fallback="none"
        )

        # Only (a0, p1) has a negative in (1.0, 1.5); (p1, a0) has none and is skipped
        assert anchors.shape == (1, 2)
        assert torch.allclose(anchors[0], embeddings[0])
        assert torch.allclose(positives[0], embeddings[1])
        assert torch.allclose(negatives[0], embeddings[3])  # n_semi selected

    def test_picks_closest_when_multiple_semi_hard_exist(self) -> None:
        # Anchor at 0 deg, positive at 60 deg (d_ap^2 = 1.0),
        # margin = 0.8 -> band (1.0, 1.8)
        # n_semi_far at 80 deg (d^2 ~= 1.653),
        # n_semi_near at 66 deg (d^2 ~= 1.187)
        a0 = _unit_vec(0.0)
        p1 = _unit_vec(math.radians(60.0))
        n_semi_far = _unit_vec(math.radians(80.0))
        n_semi_near = _unit_vec(math.radians(66.0))

        embeddings = torch.tensor([a0, p1, n_semi_far, n_semi_near], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 2])

        anchors, _, negatives = mine_semi_hard_triplets(
            embeddings, labels, margin=0.8, fallback="none"
        )

        assert anchors.shape == (1, 2)
        assert torch.allclose(negatives[0], embeddings[3])  # n_semi_near is closer to d_ap

    def test_fallback_hardest_vs_none(self) -> None:
        # Anchor at 0 deg, positive at 60 deg (d_ap^2 = 1.0),
        # margin = 0.2 -> band (1.0, 1.2)
        # Only a hard negative at 30 deg (d^2 ~= 0.268)
        # and an easy negative at 180 deg (d^2 = 4.0)
        # Distinct labels [1, 2] ensure only class 0 forms (a, p) pairs.
        a0 = _unit_vec(0.0)
        p1 = _unit_vec(math.radians(60.0))
        n_hard = _unit_vec(math.radians(30.0))
        n_easy = _unit_vec(math.radians(180.0))

        embeddings = torch.tensor([a0, p1, n_hard, n_easy], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 2])

        # With fallback="hardest", both (a0, p1) and (p1, a0) fall back to n_hard
        a_h, _p_h, n_h = mine_semi_hard_triplets(
            embeddings, labels, margin=0.2, fallback="hardest"
        )
        assert a_h.shape == (2, 2)
        assert torch.allclose(n_h[0], embeddings[2])
        assert torch.allclose(n_h[1], embeddings[2])

        # With fallback="none", zero triplets are returned
        a_n, p_n, n_n = mine_semi_hard_triplets(embeddings, labels, margin=0.2, fallback="none")
        assert a_n.shape == (0, 2)
        assert p_n.shape == (0, 2)
        assert n_n.shape == (0, 2)

    @pytest.mark.parametrize(
        "labels",
        [
            torch.tensor([0, 1, 2, 3]),  # All distinct classes (no positives)
            torch.tensor([0, 0, 0, 0]),  # Single class (no negatives)
        ],
    )
    def test_degenerate_batches_return_empty_preserving_grad(
        self, labels: torch.Tensor
    ) -> None:
        emb = F.normalize(torch.randn(4, 32), p=2, dim=1).requires_grad_(True)
        a, p, n = mine_semi_hard_triplets(emb, labels, margin=0.2)

        assert a.shape == (0, 32)
        assert p.shape == (0, 32)
        assert n.shape == (0, 32)
        assert a.dtype == emb.dtype
        assert a.requires_grad

        (a.sum() + p.sum() + n.sum()).backward()
        assert emb.grad is not None and emb.grad.shape == (4, 32)

    def test_gradient_flows_to_input_embeddings(self) -> None:
        torch.manual_seed(7)
        raw = torch.randn(8, 16, requires_grad=True)
        emb = F.normalize(raw, p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        a, p, n = mine_semi_hard_triplets(emb, labels, margin=0.3)
        loss = torch.sum((a - p) ** 2 - (a - n) ** 2)
        loss.backward()

        assert raw.grad is not None
        assert raw.grad.abs().sum() > 0.0

    def test_invalid_arguments_raise(self) -> None:
        emb = F.normalize(torch.randn(4, 8), p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1])

        with pytest.raises(ValueError, match="margin must be > 0"):
            mine_semi_hard_triplets(emb, labels, margin=0.0)

        with pytest.raises(ValueError, match="Unsupported fallback"):
            mine_semi_hard_triplets(emb, labels, fallback="random")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Expected embeddings"):
            mine_semi_hard_triplets(emb, torch.tensor([0, 1]))
