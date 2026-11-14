"""Unit tests for FaceEmbeddingNet architecture and properties."""

import pytest
import torch

from src.models import FaceEmbeddingNet


@pytest.mark.parametrize("backbone", ["resnet18", "mobilenet_v3_small"])
@pytest.mark.parametrize("dim", [64, 128])
def test_embedding_output_shape(backbone: str, dim: int) -> None:
    """Verifies that the network produces tensors of shape (B, embedding_dim)."""
    batch_size = 4
    model = FaceEmbeddingNet(backbone=backbone, embedding_dim=dim, pretrained=False)  # type: ignore[arg-type]
    x = torch.randn(batch_size, 3, 128, 128)

    out = model(x)
    assert out.shape == (batch_size, dim)


@pytest.mark.parametrize("backbone", ["resnet18", "mobilenet_v3_small"])
def test_l2_unit_norm(backbone: str) -> None:
    """Verifies all output embeddings lie strictly on the unit hypersphere."""
    model = FaceEmbeddingNet(backbone=backbone, embedding_dim=128, pretrained=False)  # type: ignore[arg-type]
    model.eval()
    x = torch.randn(8, 3, 128, 128)

    with torch.inference_mode():
        embeddings = model(x)

    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"Norms were: {norms}"


def test_single_image_eval_inference_is_deterministic() -> None:
    """Verifies batch_size=1 inference works in eval mode (required by SmartGate)."""
    model = FaceEmbeddingNet(backbone="resnet18", embedding_dim=128, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 128, 128)

    with torch.inference_mode():
        emb1 = model(x)
        emb2 = model(x)

    assert emb1.shape == (1, 128)
    assert torch.allclose(emb1, emb2, atol=1e-6)


def test_distinct_inputs_produce_distinct_embeddings() -> None:
    """Guards against embedding collapse where all inputs map to the same vector."""
    torch.manual_seed(42)
    model = FaceEmbeddingNet(backbone="resnet18", embedding_dim=128, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 128, 128)

    with torch.inference_mode():
        embeddings = model(x)

    cosine_sim = torch.dot(embeddings[0], embeddings[1]).item()
    assert cosine_sim < 0.999, f"Embeddings collapsed; cosine similarity={cosine_sim:.6f}"


def test_gradient_flow() -> None:
    """Verifies gradients backpropagate through both projection head and backbone."""
    model = FaceEmbeddingNet(
        backbone="resnet18", embedding_dim=128, pretrained=False, freeze_backbone=False
    )
    x = torch.randn(2, 3, 64, 64)

    embeddings = model(x)
    loss = embeddings.sum()
    loss.backward()

    head_grads = [
        p.grad
        for p in model.projection_head.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(head_grads) > 0
    assert any(g.abs().sum() > 0.0 for g in head_grads)

    backbone_grads = [
        p.grad for p in model.backbone.parameters() if p.requires_grad and p.grad is not None
    ]
    assert len(backbone_grads) > 0
    assert any(g.abs().sum() > 0.0 for g in backbone_grads)


def test_frozen_backbone() -> None:
    """Verifies backbone stays frozen while projection head trains when freeze_backbone=True."""
    model = FaceEmbeddingNet(
        backbone="resnet18", embedding_dim=128, pretrained=False, freeze_backbone=True
    )
    for param in model.backbone.parameters():
        assert not param.requires_grad

    head_params = list(model.projection_head.parameters())
    assert len(head_params) > 0
    assert all(p.requires_grad for p in head_params)

    # Confirm backward pass updates only the head and leaves backbone grads as None
    x = torch.randn(2, 3, 64, 64)
    model(x).sum().backward()

    assert all(p.grad is None for p in model.backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0.0 for p in head_params)


def test_invalid_backbone_raises() -> None:
    """Verifies unsupported backbone names fail fast with ValueError."""
    with pytest.raises(ValueError):
        FaceEmbeddingNet(backbone="unsupported_net", embedding_dim=128, pretrained=False)  # type: ignore[arg-type]
