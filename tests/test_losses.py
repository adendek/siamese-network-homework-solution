"""Unit tests for TripletMarginLoss."""

import pytest
import torch
import torch.nn.functional as F

from src.losses import TripletMarginLoss


class TestTripletMarginLossInit:
    @pytest.mark.parametrize("bad_margin", [0.0, -0.1, -5.0])
    def test_invalid_margin_raises(self, bad_margin: float) -> None:
        with pytest.raises(ValueError, match="margin must be > 0"):
            TripletMarginLoss(margin=bad_margin)

    @pytest.mark.parametrize("bad_reduction", ["avg", "batchmean", "", None])
    def test_invalid_reduction_raises(self, bad_reduction: str) -> None:
        with pytest.raises(ValueError, match="Unsupported reduction"):
            TripletMarginLoss(reduction=bad_reduction)  # type: ignore[arg-type]


class TestTripletMarginLossForward:
    def test_exact_values_and_reductions(self) -> None:
        # Triplet 0 (hard/active): d_ap = 1.0, d_an = 0.25, margin = 0.2 -> loss = 0.95
        # Triplet 1 (easy/inactive): d_ap = 0.25, d_an = 1.0, margin = 0.2 -> loss = 0.0
        anchors = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        positives = torch.tensor([[1.0, 0.0], [0.5, 0.0]])
        negatives = torch.tensor([[0.0, 0.5], [0.0, 1.0]])

        loss_none = TripletMarginLoss(margin=0.2, reduction="none")(
            anchors, positives, negatives
        )
        loss_mean = TripletMarginLoss(margin=0.2, reduction="mean")(
            anchors, positives, negatives
        )
        loss_sum = TripletMarginLoss(margin=0.2, reduction="sum")(anchors, positives, negatives)

        assert loss_none.shape == (2,)
        assert torch.allclose(loss_none, torch.tensor([0.95, 0.0]))
        assert torch.isclose(loss_mean, torch.tensor(0.475))
        assert torch.isclose(loss_sum, torch.tensor(0.95))

    def test_boundary_margin_gives_zero_loss(self) -> None:
        # d_ap = 0.25, d_an = 0.45, margin = 0.2 -> 0.25 - 0.45 + 0.2 = 0.0
        anchors = torch.tensor([[0.0, 0.0]])
        positives = torch.tensor([[0.5, 0.0]])
        negatives = torch.tensor([[0.0, (0.45) ** 0.5]])

        loss = TripletMarginLoss(margin=0.2, reduction="mean")(anchors, positives, negatives)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_l2_normalized_cosine_identity(self) -> None:
        # On the unit hypersphere, ||u - v||^2 == 2 - 2 * (u . v)
        torch.manual_seed(42)
        a = F.normalize(torch.randn(16, 128), p=2, dim=1)
        p = F.normalize(torch.randn(16, 128), p=2, dim=1)
        n = F.normalize(torch.randn(16, 128), p=2, dim=1)
        margin = 0.3

        d_ap_cos = 2.0 - 2.0 * (a * p).sum(dim=1)
        d_an_cos = 2.0 - 2.0 * (a * n).sum(dim=1)
        expected = F.relu(d_ap_cos - d_an_cos + margin).mean()

        actual = TripletMarginLoss(margin=margin, reduction="mean")(a, p, n)
        assert torch.allclose(actual, expected, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
    def test_dtype_preservation(self, dtype: torch.dtype) -> None:
        a = torch.randn(4, 8, dtype=dtype)
        p = torch.randn(4, 8, dtype=dtype)
        n = torch.randn(4, 8, dtype=dtype)

        loss = TripletMarginLoss(margin=0.2)(a, p, n)
        assert loss.dtype == dtype


class TestTripletMarginLossEdgeCases:
    @pytest.mark.parametrize(
        "reduction,expected_shape", [("mean", ()), ("sum", ()), ("none", (0,))]
    )
    def test_empty_triplets(self, reduction: str, expected_shape: tuple[int, ...]) -> None:
        a = torch.empty((0, 64), dtype=torch.float32, requires_grad=True)
        p = torch.empty((0, 64), dtype=torch.float32, requires_grad=True)
        n = torch.empty((0, 64), dtype=torch.float32, requires_grad=True)

        criterion = TripletMarginLoss(margin=0.2, reduction=reduction)  # type: ignore[arg-type]
        loss = criterion(a, p, n)

        assert loss.shape == expected_shape
        assert loss.dtype == torch.float32
        assert loss.device == a.device
        assert loss.requires_grad

        # Ensure backward pass works cleanly on empty reduction
        loss.sum().backward()
        assert a.grad is not None and a.grad.shape == (0, 64)

    def test_shape_mismatch_raises(self) -> None:
        criterion = TripletMarginLoss()

        # Mismatched batch dimension (prevents silent broadcasting of (1, d) vs (4, d))
        with pytest.raises(ValueError, match="identical shapes"):
            criterion(torch.randn(1, 16), torch.randn(4, 16), torch.randn(4, 16))

        # Mismatched embedding dimension
        with pytest.raises(ValueError, match="identical shapes"):
            criterion(torch.randn(4, 16), torch.randn(4, 32), torch.randn(4, 16))

        # Non-2D inputs (1D or 3D)
        with pytest.raises(ValueError, match="2-D tensors"):
            criterion(torch.randn(16), torch.randn(16), torch.randn(16))

        with pytest.raises(ValueError, match="2-D tensors"):
            criterion(torch.randn(2, 4, 16), torch.randn(2, 4, 16), torch.randn(2, 4, 16))


class TestTripletMarginLossGradients:
    def test_analytical_gradients_active_and_inactive(self) -> None:
        # Row 0 is active (d_ap=1.0, d_an=0.25, m=0.2 -> L=0.95)
        # Row 1 is inactive (d_ap=0.25, d_an=1.0, m=0.2 -> L=0.0)
        a = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
        p = torch.tensor([[1.0, 0.0], [0.5, 0.0]], requires_grad=True)
        n = torch.tensor([[0.0, 0.5], [0.0, 1.0]], requires_grad=True)

        loss = TripletMarginLoss(margin=0.2, reduction="sum")(a, p, n)
        loss.backward()

        # For active L = ||a - p||^2 - ||a - n||^2 + m:
        # dL/da = 2*(n - p), dL/dp = 2*(p - a), dL/dn = 2*(a - n)
        expected_grad_a = torch.tensor([[-2.0, 1.0], [0.0, 0.0]])
        expected_grad_p = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
        expected_grad_n = torch.tensor([[0.0, -1.0], [0.0, 0.0]])

        assert a.grad is not None and torch.allclose(a.grad, expected_grad_a)
        assert p.grad is not None and torch.allclose(p.grad, expected_grad_p)
        assert n.grad is not None and torch.allclose(n.grad, expected_grad_n)
