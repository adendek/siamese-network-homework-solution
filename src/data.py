"""Dataset loaders, image transforms, and identity-aware PK batch sampling."""

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Protocol

import torchvision.transforms as transforms
from torch.utils.data import Sampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _HasTargets(Protocol):
    """Structural protocol for datasets exposing integer class labels via `.targets`."""

    targets: Sequence[int]


def get_train_transforms(
    image_size: tuple[int, int] = (128, 128),
) -> transforms.Compose:
    """Returns data augmentation transforms for training face embeddings.

    Args:
        image_size: Target ``(height, width)`` for model input. Defaults to ``(128, 128)``.

    Returns:
        Composed torchvision transform pipeline producing normalized ``(3, H, W)`` tensors.

    Notes:
        - **Why horizontal flips are safe here:** Unlike digit or text classification, human
          faces are approximately bilaterally symmetric—a horizontally flipped face still
          belongs to the same identity, effectively doubling pose variety.
        - **Gentle photometric jitter:** ``ColorJitter(brightness=0.1, contrast=0.1)``
          simulates mild camera exposure and lighting shifts without distorting skin tones
          or facial geometry.
        - **ImageNet statistics:** Normalizing with ``IMAGENET_MEAN`` and ``IMAGENET_STD``
          aligns input pixel distributions with pretrained ``resnet18`` and
          ``mobilenet_v3_small`` backbones.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(
    image_size: tuple[int, int] = (128, 128),
) -> transforms.Compose:
    """Returns deterministic preprocessing transforms for validation, gallery, and probe.

    Args:
        image_size: Target ``(height, width)`` for model input. Defaults to ``(128, 128)``.

    Returns:
        Composed torchvision transform pipeline producing normalized ``(3, H, W)`` tensors.

    Notes:
        - **Determinism:** Random augmentations are omitted so gallery templates and probe
          queries map to reproducible points on the embedding hypersphere.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_identity_index_map(dataset: _HasTargets) -> dict[int, list[int]]:
    """Builds a mapping from identity class label to sample indices in the dataset.

    Args:
        dataset: Any dataset (such as ``torchvision.datasets.ImageFolder``) exposing
            integer identity labels in ``dataset.targets``.

    Returns:
        Dictionary mapping ``identity_id -> list[dataset_index]``.
    """
    identity_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        identity_to_indices[int(label)].append(idx)
    return dict(identity_to_indices)


class PKBatchSampler(Sampler[list[int]]):
    """Samples P distinct identities with K images each per mini-batch.

    Produces batches of size ``P * K``, guaranteeing that every anchor has both
    same-identity positives and different-identity negatives in the same batch for
    online triplet mining.

    Args:
        identity_to_indices: Mapping from identity ID to non-empty sample index list.
        p: Number of distinct identities per batch (must be ``> 1``).
        k: Number of images sampled per identity (must be ``> 1``).
        batches_per_epoch: Number of batches to yield per epoch. If ``None``,
            defaults to ``max(1, len(identity_to_indices) // p)``.
        seed: Optional random seed for reproducible batch sampling.

    Raises:
        ValueError: If ``p <= 1``, ``k <= 1``, ``batches_per_epoch <= 0``, fewer than ``p``
            identities are available, or any identity has an empty index list.

    Notes:
        - **Why standard random shuffling fails for triplet mining:** With thousands of
          identities, a uniform random batch of size 32 almost never contains two photos of
          the same person, leaving ``pos_mask`` empty and yielding zero triplets.
        - **Sampling with vs. without replacement:** If an identity has at least ``K``
          images, ``K`` distinct images are drawn without replacement. If an identity has
          fewer than ``K`` images (e.g., long-tail datasets where some people have only 1-2
          photos), images are sampled *with* replacement so the batch shape remains fixed
          at ``P * K``.
        - **Usage with PyTorch ``DataLoader``:** Pass this sampler via ``batch_sampler=...``
          (and leave ``batch_size``, ``shuffle``, and ``drop_last`` at their defaults).
    """

    def __init__(
        self,
        identity_to_indices: dict[int, list[int]],
        p: int,
        k: int,
        batches_per_epoch: int | None = None,
        seed: int | None = None,
    ) -> None:
        """Init the PKBatchSampler."""
        if p <= 1:
            raise ValueError(f"P must be > 1 to allow negative pairs, got {p}")
        if k <= 1:
            raise ValueError(f"K must be > 1 to allow positive pairs, got {k}")
        if len(identity_to_indices) < p:
            raise ValueError(
                f"Total identities ({len(identity_to_indices)}) must be >= P ({p})"
            )
        if batches_per_epoch is not None and batches_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch must be > 0 when specified, got {batches_per_epoch}"
            )

        empty_ids = [ident for ident, idxs in identity_to_indices.items() if len(idxs) == 0]
        if empty_ids:
            raise ValueError(f"Found identities with empty index lists: {empty_ids}")

        self.identity_to_indices = identity_to_indices
        self.identities = list(identity_to_indices.keys())
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.batches_per_epoch = (
            batches_per_epoch
            if batches_per_epoch is not None
            else max(1, len(self.identities) // p)
        )
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        """Returns the total number of batches in one epoch."""
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yields batches of dataset indices of length ``P * K``."""
        for _ in range(self.batches_per_epoch):
            batch_indices: list[int] = []

            # Sample P distinct identities without replacement
            selected_identities = self.rng.sample(self.identities, self.p)

            for identity in selected_identities:
                available_indices = self.identity_to_indices[identity]

                if len(available_indices) >= self.k:
                    sampled = self.rng.sample(available_indices, self.k)
                else:
                    sampled = self.rng.choices(available_indices, k=self.k)

                batch_indices.extend(sampled)

            yield batch_indices
