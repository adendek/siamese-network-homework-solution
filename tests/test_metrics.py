"""Unit and integration tests for verification pair generation and biometric metrics."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.metrics import (
    VerificationMetrics,
    evaluate_verification,
    generate_evaluation_pairs,
)


class TestGenerateEvaluationPairs:
    def test_balanced_output_shape_and_dtypes(self) -> None:
        torch.manual_seed(42)
        embeddings = F.normalize(torch.randn(30, 64), p=2, dim=1)
        # 6 identities x 5 samples each
        labels = torch.tensor([i for i in range(6) for _ in range(5)])

        distances, actual_matches = generate_evaluation_pairs(
            embeddings, labels, num_pairs=100, seed=42
        )

        assert distances.shape == (100,)
        assert actual_matches.shape == (100,)
        assert distances.dtype == np.float64
        assert actual_matches.dtype == np.int64
        assert int((actual_matches == 1).sum()) == 50
        assert int((actual_matches == 0).sum()) == 50
        assert np.all(distances >= 0.0)
        assert np.all(distances <= 4.0 + 1e-5)

    def test_exact_geometric_distances_and_no_self_pairs(self) -> None:
        # Identity 0 clusters tightly near [1, 0]; Identity 1 clusters near [0, 1].
        # Distinct samples within each identity ensure self-pairing (dist == 0.0) never occurs.
        emb = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.01],
                    [1.0, -0.01],
                    [0.01, 1.0],
                    [-0.01, 1.0],
                ],
                dtype=torch.float32,
            ),
            p=2,
            dim=1,
        )
        labels = torch.tensor([0, 0, 1, 1])

        distances, actual_matches = generate_evaluation_pairs(emb, labels, num_pairs=40, seed=7)

        pos_dists = distances[actual_matches == 1]
        neg_dists = distances[actual_matches == 0]

        # Positive pairs must come from distinct images (dist > 0) and stay near 0
        assert np.all(pos_dists > 0.0)
        assert np.all(pos_dists < 1e-3)
        # Negative pairs are nearly orthogonal (squared distance ~= 2.0)
        assert np.allclose(neg_dists, 2.0, atol=0.05)

    def test_seed_determinism_and_grad_tensor_handling(self) -> None:
        torch.manual_seed(11)
        embeddings = F.normalize(torch.randn(20, 32), p=2, dim=1).requires_grad_(True)
        labels = torch.tensor([i for i in range(4) for _ in range(5)])

        d1, m1 = generate_evaluation_pairs(embeddings, labels, num_pairs=50, seed=123)
        d2, m2 = generate_evaluation_pairs(embeddings, labels, num_pairs=50, seed=123)
        d3, _ = generate_evaluation_pairs(embeddings, labels, num_pairs=50, seed=999)

        assert np.array_equal(d1, d2)
        assert np.array_equal(m1, m2)
        assert not np.array_equal(d1, d3)

    @pytest.mark.parametrize("bad_num_pairs", [0, 1, -4, 15])
    def test_invalid_num_pairs_raises(self, bad_num_pairs: int) -> None:
        emb = torch.randn(6, 8)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        with pytest.raises(ValueError, match="num_pairs must be an even integer >= 2"):
            generate_evaluation_pairs(emb, labels, num_pairs=bad_num_pairs)

    def test_insufficient_identities_or_positives_raises(self) -> None:
        emb = torch.randn(4, 8)

        # Only 1 unique identity -> cannot form negative pairs
        with pytest.raises(ValueError, match="at least 2 distinct identities"):
            generate_evaluation_pairs(emb, torch.tensor([0, 0, 0, 0]), num_pairs=4)

        # All unique identities -> no identity has >= 2 samples for positive pairs
        with pytest.raises(ValueError, match="at least one identity with >= 2 samples"):
            generate_evaluation_pairs(emb, torch.tensor([0, 1, 2, 3]), num_pairs=4)

    def test_mismatched_shapes_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected embeddings"):
            generate_evaluation_pairs(torch.randn(4, 8), torch.tensor([0, 0, 1]), num_pairs=4)

        with pytest.raises(ValueError, match="Expected embeddings"):
            generate_evaluation_pairs(torch.randn(4), torch.tensor([0, 0, 1, 1]), num_pairs=4)


class TestEvaluateVerification:
    def test_perfect_separation(self) -> None:
        distances = np.array([0.05, 0.10, 0.15, 1.80, 2.00, 2.20])
        actual_matches = np.array([1, 1, 1, 0, 0, 0])

        metrics = evaluate_verification(distances, actual_matches)

        assert isinstance(metrics, VerificationMetrics)
        assert metrics.auc == pytest.approx(1.0)
        assert metrics.eer == pytest.approx(0.0)
        assert metrics.best_accuracy == pytest.approx(1.0)
        assert 0.15 <= metrics.best_threshold < 1.80
        assert metrics.mean_pos_dist == pytest.approx(0.10)
        assert metrics.mean_neg_dist == pytest.approx(2.00)

    def test_completely_inverted_predictions(self) -> None:
        # Impostors are closer than genuine pairs -> worst possible classifier
        distances = np.array([2.5, 3.0, 0.1, 0.2])
        actual_matches = np.array([1, 1, 0, 0])

        metrics = evaluate_verification(distances, actual_matches)

        assert metrics.auc == pytest.approx(0.0)
        assert metrics.eer == pytest.approx(1.0)

    def test_known_analytical_partial_overlap(self) -> None:
        # Sorted order: 0.1(P), 0.2(P), 0.3(N), 0.4(P), 0.5(N), 0.7(N), 0.8(P), 0.9(N)
        # EER = 0.25 occurs uniquely at tau = 0.4 (FPR = 1/4, FNR = 1/4).
        # Max accuracy = 0.75 (6/8) is tied at tau = 0.2 (TP=2, TN=4) and tau = 0.4 (TP=3, TN=3).
        distances = np.array([0.1, 0.2, 0.4, 0.8, 0.3, 0.5, 0.7, 0.9])
        actual_matches = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        metrics = evaluate_verification(distances, actual_matches)

        assert metrics.auc == pytest.approx(0.75)
        assert metrics.eer == pytest.approx(0.25)
        assert metrics.best_accuracy == pytest.approx(0.75)
        assert (0.2 <= metrics.best_threshold < 0.3) or (0.4 <= metrics.best_threshold < 0.5)
        assert metrics.mean_pos_dist == pytest.approx(0.375)
        assert metrics.mean_neg_dist == pytest.approx(0.60)

    def test_non_unit_distance_scale_above_four(self) -> None:
        # Ensures threshold search adapts when distances exceed [0.0, 4.0]
        distances = np.array([10.0, 12.0, 50.0, 60.0])
        actual_matches = np.array([1, 1, 0, 0])

        metrics = evaluate_verification(distances, actual_matches)

        assert metrics.best_accuracy == pytest.approx(1.0)
        assert 12.0 <= metrics.best_threshold < 50.0

    def test_invalid_inputs_raise(self) -> None:
        # Empty arrays
        with pytest.raises(ValueError, match="non-empty 1-D arrays"):
            evaluate_verification(np.array([]), np.array([]))

        # Shape mismatch
        with pytest.raises(ValueError, match="non-empty 1-D arrays"):
            evaluate_verification(np.array([0.1, 0.2]), np.array([1]))

        # Single class only (all positives or all negatives)
        with pytest.raises(ValueError, match="both positive .* and negative"):
            evaluate_verification(np.array([0.1, 0.2]), np.array([1, 1]))

        with pytest.raises(ValueError, match=r"both positive .* and negative"):
            evaluate_verification(np.array([0.1, 0.2]), np.array([0, 0]))


class TestMetricsEndToEnd:
    def test_pair_generation_to_evaluation_pipeline(self) -> None:
        torch.manual_seed(0)
        # Simulate 5 well-separated class centroids with slight intra-class noise
        centroids = F.normalize(torch.randn(5, 32), p=2, dim=1)
        emb_list = []
        lbl_list = []
        for class_id in range(5):
            for _ in range(6):
                noisy = F.normalize(centroids[class_id] + 0.05 * torch.randn(32), p=2, dim=0)
                emb_list.append(noisy)
                lbl_list.append(class_id)

        embeddings = torch.stack(emb_list)
        labels = torch.tensor(lbl_list)

        distances, matches = generate_evaluation_pairs(
            embeddings, labels, num_pairs=200, seed=42
        )
        metrics = evaluate_verification(distances, matches)

        assert metrics.auc > 0.99
        assert metrics.eer < 0.02
        assert metrics.best_accuracy > 0.98
        assert metrics.mean_pos_dist < metrics.mean_neg_dist
