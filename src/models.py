"""Embedding models for face recognition and deep metric learning."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FaceEmbeddingNet(nn.Module):
    """Backbone neural network with projection head for hypersphere metric embeddings.

    Adapts a vision backbone (e.g. ResNet, MobileNet) to output an L2-normalized
    d-dimensional feature vector on the unit hypersphere.

    Attributes:
        backbone_name: Name of the convolutional architecture.
        embedding_dim: Dimensionality of the target metric space (d).
        backbone: The feature extraction network.
        projection_head: Linear layer mapping backbone features to embedding_dim.
    """

    def __init__(
        self,
        backbone: Literal["resnet18", "mobilenet_v3_small"] = "resnet18",
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """Initializes FaceEmbeddingNet.

        Args:
            backbone: Choice of vision backbone ('resnet18' or 'mobilenet_v3_small').
            embedding_dim: Dimension of the output embedding vector.
            pretrained: If True, initializes with ImageNet pre-trained weights.
            freeze_backbone: If True, freezes backbone layers and trains only the head.
        """
        super().__init__()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            # Remove classification head
            model.fc = nn.Identity()
            self.backbone = model

        elif backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            mb_model = models.mobilenet_v3_small(weights=weights)
            in_features = mb_model.classifier[0].out_features
            # Replace final linear head with identity
            mb_model.classifier[-1] = nn.Identity()
            self.backbone = mb_model
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head: maps extracted feature vector to metric space
        self.projection_head = nn.Linear(in_features, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts and L2-normalizes face embeddings.

        Args:
            x: Input face images of shape (B, 3, H, W).

        Returns:
            Unit-normalized embeddings of shape (B, embedding_dim).
        """
        # Feature extraction
        features = self.backbone(x)

        # Dimensional projection
        embeddings = self.projection_head(features)

        # Hypersphere projection: ||z||_2 = 1.0
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12)

        return normalized_embeddings
