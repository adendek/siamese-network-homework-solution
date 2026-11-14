"""Training pipeline for deep metric face embedding learning."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import evaluate_verification, generate_evaluation_pairs
from src.mining import mine_semi_hard_triplets


@dataclass
class EpochStats:
    """Container for per-epoch training and validation metrics.

    Attributes:
        train_loss: Average triplet margin loss across processed training batches.
        active_triplets_ratio: Fraction in ``[0.0, 1.0]`` of mined triplets that violated
            the margin (``d(a, p) - d(a, n) + margin > 0``).
        val_accuracy: Best 1:1 verification accuracy on validation pairs.
        val_auc: ROC Area Under the Curve on validation pairs.
        val_eer: Equal Error Rate on validation pairs (lower is better).
        val_margin: Separation gap ``mean_neg_dist - mean_pos_dist`` on validation pairs.
        epoch: 1-based epoch index.
        is_best: ``True`` if this epoch achieved the best monitored validation score so far.
    """

    train_loss: float
    active_triplets_ratio: float
    val_accuracy: float
    val_auc: float
    val_eer: float
    val_margin: float
    epoch: int = 0
    is_best: bool = False


class MetricTrainer:
    """Coordinates model training, online triplet mining, optimization, and validation.

    Args:
        model: Embedding network mapping ``(B, C, H, W)`` images to L2-normalized
            ``(B, d)`` embeddings. If the model exposes ``.backbone`` and
            ``.projection_head`` submodules, differential learning rates are applied.
        criterion: Triplet loss module taking ``(anchors, positives, negatives)``.
        train_loader: Training ``DataLoader`` (typically backed by ``PKBatchSampler``).
        val_loader: Validation ``DataLoader`` over held-out identities.
        device: Target compute device (e.g., ``torch.device('cuda')``, ``'mps'``, or ``'cpu'``).
        margin: Triplet separation margin ``alpha > 0``. If ``None``, inferred from
            ``criterion.margin`` (defaulting to ``0.2``).
        backbone_lr: Learning rate for pretrained backbone layers. Defaults to ``1e-5``.
        head_lr: Learning rate for the randomly initialized projection head.
            Defaults to ``5e-4``.
        weight_decay: AdamW weight decay coefficient. Defaults to ``1e-4``.
        epochs: Total number of training epochs. Defaults to ``15``.
        val_num_pairs: Number of balanced positive/negative pairs sampled during
            ``validate()``. Defaults to ``1000``.
        val_seed: Deterministic seed used when sampling validation pairs. Defaults to ``42``.
        mining_fallback: Fallback strategy passed to ``mine_semi_hard_triplets``
            (``'hardest'`` or ``'none'``). Defaults to ``'hardest'``.
        checkpoint_dir: Optional directory path where ``best_model.pt`` and
            ``last_checkpoint.pt`` are saved. If ``None``, checkpoints are not written to disk.
        monitor: Validation metric used to select the best checkpoint (``'eer'`` minimizes
            Equal Error Rate; ``'auc'`` maximizes ROC-AUC). Defaults to ``'eer'``.

    Raises:
        ValueError: If ``epochs <= 0``, ``margin <= 0``, or ``margin`` disagrees with
            ``criterion.margin``.

    Notes:
        - **Why differential learning rates are used:** Pretrained backbones (ResNet18,
          MobileNetV3) already encode rich low-level facial edges and textures; updating
          them with a small LR (``1e-5``) prevents catastrophic forgetting, while the
          newly initialized linear projection head needs a larger LR (``5e-4``) to align
          features onto the unit hypersphere.
        - **Why ``active_triplets_ratio`` decreases over training:** Early in training,
          embeddings are poorly separated and most mined triplets violate the margin
          (ratio near ``1.0``). As clusters separate, fewer triplets produce non-zero
          loss, indicating convergence.
        - **Best-checkpoint restoration:** At the end of ``fit()`` (including if stopped
          early via ``KeyboardInterrupt``), the best validation weights are automatically
          reloaded into ``self.model`` when ``checkpoint_dir`` is set.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: torch.device,
        margin: float | None = 0.2,
        backbone_lr: float = 1e-5,
        head_lr: float = 5e-4,
        weight_decay: float = 1e-4,
        epochs: int = 15,
        val_num_pairs: int = 1000,
        val_seed: int = 42,
        mining_fallback: Literal["hardest", "none"] = "hardest",
        checkpoint_dir: str | Path | None = None,
        monitor: Literal["eer", "auc"] = "eer",
    ) -> None:
        """Initialize ``Trainer``."""
        if epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {epochs}")
        if monitor not in ("eer", "auc"):
            raise ValueError(
                f"Unsupported monitor metric: '{monitor}'. Expected 'eer' or 'auc'."
            )

        criterion_margin = getattr(criterion, "margin", None)
        if margin is None:
            resolved_margin = float(criterion_margin) if criterion_margin is not None else 0.2
        else:
            resolved_margin = float(margin)
            if criterion_margin is not None and not math.isclose(
                resolved_margin, float(criterion_margin), rel_tol=1e-6
            ):
                raise ValueError(
                    f"Margin mismatch: MetricTrainer received margin={resolved_margin}, "
                    f"but criterion has margin={criterion_margin}."
                )

        if resolved_margin <= 0:
            raise ValueError(f"margin must be > 0, got {resolved_margin}")

        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.margin = resolved_margin
        self.epochs = epochs
        self.val_num_pairs = val_num_pairs
        self.val_seed = val_seed
        self.mining_fallback = mining_fallback

        # Differential learning rates when backbone and projection_head are exposed
        if hasattr(self.model, "backbone") and hasattr(self.model, "projection_head"):
            backbone_params = [
                p
                for p in self.model.backbone.parameters()  # type: ignore[union-attr]
                if p.requires_grad
            ]
            head_params = [
                p
                for p in self.model.projection_head.parameters()  # type: ignore[union-attr]
                if p.requires_grad
            ]
            param_groups: list[dict[str, Any]] = []
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": backbone_lr})
            if head_params:
                param_groups.append({"params": head_params, "lr": head_lr})
            if not param_groups:
                raise ValueError("Model has no trainable parameters (all requires_grad=False).")
            self.optimizer = AdamW(param_groups, weight_decay=weight_decay)
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                raise ValueError("Model has no trainable parameters (all requires_grad=False).")
            self.optimizer = AdamW(trainable_params, lr=head_lr, weight_decay=weight_decay)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.best_score = float("inf") if monitor == "eer" else float("-inf")
        self.best_epoch: int | None = None
        self.history: list[EpochStats] = []

    def _is_improvement(self, stats: EpochStats) -> bool:
        score = stats.val_eer if self.monitor == "eer" else stats.val_auc
        if self.monitor == "eer":
            return score < self.best_score
        return score > self.best_score

    def save_checkpoint(self, filename: str, epoch: int, stats: EpochStats) -> Path:
        """Saves model, optimizer, scheduler, and epoch metrics to ``checkpoint_dir``."""
        if self.checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is None; cannot save checkpoints.")
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_score": self.best_score,
                "monitor": self.monitor,
                "stats": stats,
            },
            path,
        )
        return path

    def load_checkpoint(self, checkpoint_path: str | Path, load_optimizer: bool = True) -> int:
        """Restores model (and optionally optimizer/scheduler) state from disk."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if load_optimizer and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_score = ckpt.get("best_score", self.best_score)
        return int(ckpt.get("epoch", 0))

    def train_epoch(self) -> tuple[float, float]:
        """Runs one training epoch with online semi-hard negative mining.

        Returns:
            Tuple ``(avg_loss, active_triplets_ratio)`` across processed batches.
        """
        self.model.train()
        total_loss = 0.0
        total_active_triplets = 0
        total_mined_triplets = 0
        processed_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # 1. Forward pass: L2-normalized embeddings
            embeddings = self.model(images)

            # 2. Mine semi-hard triplets online
            anchors, positives, negatives = mine_semi_hard_triplets(
                embeddings,
                labels,
                margin=self.margin,
                fallback=self.mining_fallback,
            )

            if anchors.numel() == 0:
                continue

            # 3. Triplet margin loss computation
            loss = self.criterion(anchors, positives, negatives)

            # 4. Backward pass & optimization
            loss.backward()
            self.optimizer.step()

            # Count triplets that strictly violated the margin (loss > 0)
            with torch.no_grad():
                d_ap = torch.sum((anchors - positives) ** 2, dim=1)
                d_an = torch.sum((anchors - negatives) ** 2, dim=1)
                active_count = int(((d_ap - d_an + self.margin) > 0.0).sum().item())

            batch_loss = float(loss.item())
            num_mined = int(anchors.size(0))
            total_loss += batch_loss
            total_active_triplets += active_count
            total_mined_triplets += num_mined
            processed_batches += 1

            batch_active_ratio = active_count / max(1, num_mined)
            pbar.set_postfix(
                {
                    "loss": f"{batch_loss:.4f}",
                    "active": f"{batch_active_ratio:.1%}",
                }
            )

        avg_loss = total_loss / max(1, processed_batches)
        active_ratio = total_active_triplets / max(1, total_mined_triplets)
        return avg_loss, active_ratio

    @torch.inference_mode()
    def validate(self) -> EpochStats:
        """Evaluates 1:1 face verification metrics on held-out validation identities."""
        self.model.eval()
        all_embeddings: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            embeddings = self.model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

        cat_embeddings = torch.cat(all_embeddings, dim=0)
        cat_labels = torch.cat(all_labels, dim=0)

        distances, targets = generate_evaluation_pairs(
            cat_embeddings,
            cat_labels,
            num_pairs=self.val_num_pairs,
            seed=self.val_seed,
        )
        metrics = evaluate_verification(distances, targets)
        val_margin = metrics.mean_neg_dist - metrics.mean_pos_dist

        return EpochStats(
            train_loss=0.0,
            active_triplets_ratio=0.0,
            val_accuracy=metrics.best_accuracy,
            val_auc=metrics.auc,
            val_eer=metrics.eer,
            val_margin=val_margin,
        )

    def fit(self, resume_from: str | Path | None = None) -> list[EpochStats]:
        """Runs the full training loop across epochs with validation and checkpointing.

        Args:
            resume_from: Optional path to a saved checkpoint to resume training from.

        Returns:
            List of ``EpochStats`` recorded at the end of each epoch.
        """
        start_epoch = 1
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from, load_optimizer=True) + 1

        try:
            for epoch in range(start_epoch, self.epochs + 1):
                train_loss, active_ratio = self.train_epoch()
                self.scheduler.step()

                stats = self.validate()
                stats.epoch = epoch
                stats.train_loss = train_loss
                stats.active_triplets_ratio = active_ratio

                if self._is_improvement(stats):
                    self.best_score = stats.val_eer if self.monitor == "eer" else stats.val_auc
                    self.best_epoch = epoch
                    stats.is_best = True
                    if self.checkpoint_dir is not None:
                        self.save_checkpoint("best_model.pt", epoch, stats)

                if self.checkpoint_dir is not None:
                    self.save_checkpoint("last_checkpoint.pt", epoch, stats)

                self.history.append(stats)

                best_tag = "  <-- BEST" if stats.is_best else ""
                print(
                    f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                    f"Loss: {stats.train_loss:.4f} | "
                    f"Active Triplets: {stats.active_triplets_ratio:.1%} | "
                    f"Val Acc: {stats.val_accuracy:.2%} | "
                    f"Val AUC: {stats.val_auc:.4f} | "
                    f"EER: {stats.val_eer:.2%} | "
                    f"Margin (Neg-Pos): {stats.val_margin:.3f}{best_tag}"
                )
        except KeyboardInterrupt:
            print(
                f"\nTraining interrupted. Best epoch so far: {self.best_epoch} "
                f"({self.monitor}={self.best_score:.4f})."
            )

        if self.checkpoint_dir is not None and self.best_epoch is not None:
            best_path = self.checkpoint_dir / "best_model.pt"
            if best_path.exists():
                self.load_checkpoint(best_path, load_optimizer=False)
                print(f"Loaded best weights from epoch {self.best_epoch} ({best_path}).")

        return self.history
