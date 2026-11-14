"""Unit and integration tests for transforms, index mapping, and PKBatchSampler."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PKBatchSampler,
    build_identity_index_map,
    get_eval_transforms,
    get_train_transforms,
)


@dataclass
class _DummyDataset:
    targets: list[int]


class TestTransforms:
    @pytest.mark.parametrize("image_size", [(128, 128), (64, 96)])
    def test_train_and_eval_output_shape_and_dtype(self, image_size: tuple[int, int]) -> None:
        img = Image.new("RGB", (200, 150), color=(120, 180, 240))

        train_t = get_train_transforms(image_size=image_size)(img)
        eval_t = get_eval_transforms(image_size=image_size)(img)

        expected_shape = (3, image_size[0], image_size[1])
        assert train_t.shape == expected_shape
        assert eval_t.shape == expected_shape
        assert train_t.dtype == torch.float32
        assert eval_t.dtype == torch.float32

    def test_eval_transforms_are_deterministic(self) -> None:
        img = Image.effect_mandelbrot((160, 160), (-2, -1.5, 1, 1.5), 50).convert("RGB")
        tf = get_eval_transforms((128, 128))

        out1 = tf(img)
        out2 = tf(img)
        assert torch.equal(out1, out2)

    def test_train_transforms_apply_stochastic_augmentation(self) -> None:
        # Asymmetric gradient image so horizontal flips and jitter produce distinct tensors
        img = Image.linear_gradient("L").convert("RGB").resize((128, 128))
        tf = get_train_transforms((128, 128))

        outputs = [tf(img) for _ in range(12)]
        any_different = any(not torch.equal(outputs[0], other) for other in outputs[1:])
        assert any_different

    def test_imagenet_normalization_values(self) -> None:
        # Pure black (0.0) -> -mean / std; pure white (1.0) -> (1.0 - mean) / std
        black = Image.new("RGB", (16, 16), color=(0, 0, 0))
        white = Image.new("RGB", (16, 16), color=(255, 255, 255))
        tf = get_eval_transforms((16, 16))

        black_t = tf(black)
        white_t = tf(white)

        for c in range(3):
            expected_black = -IMAGENET_MEAN[c] / IMAGENET_STD[c]
            expected_white = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            assert torch.allclose(black_t[c], torch.full((16, 16), expected_black), atol=1e-5)
            assert torch.allclose(white_t[c], torch.full((16, 16), expected_white), atol=1e-5)


class TestBuildIdentityIndexMap:
    def test_groups_indices_by_identity(self) -> None:
        ds = _DummyDataset(targets=[0, 2, 0, 1, 2, 0])
        mapping = build_identity_index_map(ds)

        assert mapping == {
            0: [0, 2, 5],
            1: [3],
            2: [1, 4],
        }

    def test_empty_dataset_returns_empty_dict(self) -> None:
        ds = _DummyDataset(targets=[])
        assert build_identity_index_map(ds) == {}

    def test_non_contiguous_labels(self) -> None:
        ds = _DummyDataset(targets=[10, 42, 10])
        assert build_identity_index_map(ds) == {10: [0, 2], 42: [1]}


class TestPKBatchSamplerInit:
    @pytest.mark.parametrize("bad_p", [1, 0, -3])
    def test_invalid_p_raises(self, bad_p: int) -> None:
        mapping = {0: [0, 1], 1: [2, 3]}
        with pytest.raises(ValueError, match="P must be > 1"):
            PKBatchSampler(mapping, p=bad_p, k=2)

    @pytest.mark.parametrize("bad_k", [1, 0, -2])
    def test_invalid_k_raises(self, bad_k: int) -> None:
        mapping = {0: [0, 1], 1: [2, 3]}
        with pytest.raises(ValueError, match="K must be > 1"):
            PKBatchSampler(mapping, p=2, k=bad_k)

    def test_fewer_identities_than_p_raises(self) -> None:
        mapping = {0: [0, 1], 1: [2, 3]}
        with pytest.raises(ValueError, match="Total identities"):
            PKBatchSampler(mapping, p=3, k=2)

    @pytest.mark.parametrize("bad_batches", [0, -1])
    def test_non_positive_batches_per_epoch_raises(self, bad_batches: int) -> None:
        mapping = {0: [0, 1], 1: [2, 3]}
        with pytest.raises(ValueError, match="batches_per_epoch must be > 0"):
            PKBatchSampler(mapping, p=2, k=2, batches_per_epoch=bad_batches)

    def test_empty_identity_index_list_raises(self) -> None:
        mapping = {0: [0, 1], 1: []}
        with pytest.raises(ValueError, match="empty index lists"):
            PKBatchSampler(mapping, p=2, k=2)

    def test_default_batches_per_epoch(self) -> None:
        mapping = {i: [i * 2, i * 2 + 1] for i in range(10)}
        sampler = PKBatchSampler(mapping, p=4, k=2)
        assert len(sampler) == 2  # 10 // 4 == 2
        assert sampler.batch_size == 8


class TestPKBatchSamplerSampling:
    def test_strict_pk_invariant_per_batch(self) -> None:
        # 6 identities, 5 images each (indices 0..29)
        mapping = {ident: list(range(ident * 5, (ident + 1) * 5)) for ident in range(6)}
        idx_to_ident = {idx: ident for ident, idxs in mapping.items() for idx in idxs}

        p, k = 3, 4
        sampler = PKBatchSampler(mapping, p=p, k=k, batches_per_epoch=10, seed=42)
        batches = list(sampler)

        assert len(batches) == 10
        for batch in batches:
            assert len(batch) == p * k
            identities_in_batch = [idx_to_ident[idx] for idx in batch]
            counts = Counter(identities_in_batch)

            # Exactly P distinct identities, each appearing exactly K times
            assert len(counts) == p
            assert set(counts.values()) == {k}

    def test_samples_without_replacement_when_sufficient_images(self) -> None:
        # Each identity has exactly K=4 images -> all indices in a batch must be unique
        mapping = {ident: list(range(ident * 4, (ident + 1) * 4)) for ident in range(4)}
        sampler = PKBatchSampler(mapping, p=2, k=4, batches_per_epoch=8, seed=123)

        for batch in sampler:
            assert len(batch) == 8
            assert len(set(batch)) == 8  # No duplicate indices

    def test_samples_with_replacement_when_fewer_than_k_images(self) -> None:
        # Identity 0 has only 1 image; identity 1 has 2 images; K=4 requires replacement
        mapping = {0: [10], 1: [20, 21]}
        sampler = PKBatchSampler(mapping, p=2, k=4, batches_per_epoch=5, seed=7)

        for batch in sampler:
            assert len(batch) == 8
            assert batch.count(10) == 4
            assert sum(1 for idx in batch if idx in (20, 21)) == 4

    def test_seed_reproducibility_and_epoch_progression(self) -> None:
        mapping = {i: list(range(i * 5, (i + 1) * 5)) for i in range(8)}

        s1 = PKBatchSampler(mapping, p=3, k=2, batches_per_epoch=5, seed=99)
        s2 = PKBatchSampler(mapping, p=3, k=2, batches_per_epoch=5, seed=99)
        s_diff = PKBatchSampler(mapping, p=3, k=2, batches_per_epoch=5, seed=100)

        epoch1_s1 = list(s1)
        epoch1_s2 = list(s2)
        epoch1_diff = list(s_diff)
        epoch2_s1 = list(s1)

        assert epoch1_s1 == epoch1_s2
        assert epoch1_s1 != epoch1_diff
        # Subsequent epochs should continue sampling fresh batches rather than repeating
        assert epoch1_s1 != epoch2_s1


class TestDataLoaderIntegration:
    def test_pk_sampler_with_pytorch_dataloader(self) -> None:
        # 5 identities x 4 samples = 20 samples
        labels = torch.tensor([i for i in range(5) for _ in range(4)])
        images = torch.randn(20, 3, 16, 16)
        dataset = TensorDataset(images, labels)

        mapping = build_identity_index_map(_DummyDataset(targets=labels.tolist()))
        sampler = PKBatchSampler(mapping, p=3, k=2, batches_per_epoch=4, seed=42)
        loader = DataLoader(dataset, batch_sampler=sampler)

        batch_count = 0
        for batch_imgs, batch_labels in loader:
            batch_count += 1
            assert batch_imgs.shape == (6, 3, 16, 16)
            assert batch_labels.shape == (6,)
            unique_labels, counts = torch.unique(batch_labels, return_counts=True)
            assert len(unique_labels) == 3
            assert torch.all(counts == 2)

        assert batch_count == 4

    def test_end_to_end_imagefolder_pipeline(self, tmp_path: Path) -> None:
        # Create minimal ImageFolder directory structure on disk: 3 identities x 3 images
        for person in ("alice", "bob", "charlie"):
            person_dir = tmp_path / person
            person_dir.mkdir()
            for idx in range(3):
                img = Image.new("RGB", (32, 32), color=(idx * 40, 100, 150))
                img.save(person_dir / f"{person}_{idx}.jpg")

        dataset = datasets.ImageFolder(root=tmp_path, transform=get_train_transforms((64, 64)))
        mapping = build_identity_index_map(dataset)
        sampler = PKBatchSampler(mapping, p=2, k=3, batches_per_epoch=2, seed=42)
        loader = DataLoader(dataset, batch_sampler=sampler)

        batch_imgs, batch_labels = next(iter(loader))
        assert batch_imgs.shape == (6, 3, 64, 64)
        unique_labels, counts = torch.unique(batch_labels, return_counts=True)
        assert len(unique_labels) == 2
        assert torch.all(counts == 3)
