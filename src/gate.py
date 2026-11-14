"""Automated biometric access gate simulation and threshold calibration."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from src.data import get_eval_transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class GateDecision:
    """Represents the access verification result for a single query face.

    Attributes:
        access_granted: ``True`` if ``min_distance <= threshold``.
        matched_identity: Identity string of the closest enrolled person when granted,
            otherwise ``None``.
        min_distance: Minimum squared Euclidean distance ``||z_query - z_gallery||^2``
            across all enrolled gallery templates.
        threshold: Squared Euclidean distance threshold ``tau`` used for the decision.
        similarity_score: Cosine similarity in ``[-1.0, 1.0]`` derived from
            ``min_distance`` via ``1.0 - 0.5 * min_distance``.
    """

    access_granted: bool
    matched_identity: str | None
    min_distance: float
    threshold: float
    similarity_score: float


@dataclass
class GateEvaluationResult:
    """Metrics from simulating the gate against authorized and impostor probe sets.

    Attributes:
        threshold: Squared Euclidean distance threshold ``tau`` evaluated.
        total_authorized: Number of authorized probe images evaluated.
        total_impostors: Number of impostor probe images evaluated.
        true_accepts: Authorized probes granted access.
        false_rejects: Authorized probes wrongly denied access.
        true_rejects: Impostor probes properly denied access.
        false_accepts: Impostor probes wrongly granted access.
        tar: True Acceptance Rate (``true_accepts / total_authorized == 1 - frr``).
        far: False Acceptance Rate (``false_accepts / total_impostors``).
        frr: False Rejection Rate (``false_rejects / total_authorized``).
        accuracy: Overall raw classification accuracy across all probes.
    """

    threshold: float
    total_authorized: int
    total_impostors: int
    true_accepts: int
    false_rejects: int
    true_rejects: int
    false_accepts: int
    tar: float
    far: float
    frr: float
    accuracy: float

    @property
    def balanced_accuracy(self) -> float:
        """Class-balanced accuracy ``0.5 * (TAR + (1 - FAR))`` resilient to probe skew."""
        return 0.5 * (self.tar + (1.0 - self.far))


class SmartGate:
    """Simulates an automated 1:N open-set access gate using deep face embeddings.

    Args:
        model: Trained embedding network mapping ``(B, 3, H, W)`` images to L2-normalized
            ``(B, d)`` vectors.
        threshold: Squared Euclidean distance threshold ``tau`` in ``[0.0, 4.0]``.
            Queries with ``min_distance <= tau`` are granted access. Defaults to ``0.8``.
        transform: Deterministic image preprocessing pipeline. Defaults to
            ``get_eval_transforms()``.
        device: Target compute device. If ``None``, auto-selects CUDA, MPS, or CPU.

    Notes:
        - **1:N Open-Set Identification vs. 1:1 Verification:** Unlike 1:1 pair evaluation
          (which compares one query against one template), ``SmartGate`` computes the
          minimum distance across *all* ``G`` enrolled staff centroids. Taking
          ``min_{g in G} D(q, g)`` increases impostor collision risk as ``G`` grows, so
          ``threshold`` should be calibrated on a validation gallery/probe split rather
          than copied directly from 1:1 ROC curves.
        - **Multi-photo centroid enrollment:** When multiple enrollment photos exist for
          one person, ``enroll_person`` averages their unit embeddings and re-normalizes
          the mean vector back onto the unit hypersphere, reducing sensitivity to single-photo
          pose or lighting noise.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.8,
        transform: transforms.Compose | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Init the SmartGate object."""
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device).eval()
        self.threshold = float(threshold)
        self.transform = transform if transform is not None else get_eval_transforms()

        self.gallery_embeddings: torch.Tensor | None = None
        self.gallery_identities: list[str] = []

    def _prepare_tensor(
        self, image_input: str | Path | Image.Image | torch.Tensor
    ) -> torch.Tensor:
        """Converts an image path, PIL Image, or tensor into a ``(1, C, H, W)`` model input."""
        if isinstance(image_input, str | Path):
            with Image.open(image_input) as img:
                transformed = self.transform(img.convert("RGB"))
                if not isinstance(transformed, torch.Tensor):
                    raise TypeError(
                        f"Expected transform to return torch.Tensor, got {type(transformed)}."
                    )
                return transformed.unsqueeze(0)
        if isinstance(image_input, Image.Image):
            transformed = self.transform(image_input.convert("RGB"))
            if not isinstance(transformed, torch.Tensor):
                raise TypeError(
                    f"Expected transform to return torch.Tensor, got {type(transformed)}."
                )
            return transformed.unsqueeze(0)
        if isinstance(image_input, torch.Tensor):
            if image_input.ndim == 3:
                return image_input.unsqueeze(0)
            if image_input.ndim == 4:
                return image_input
            raise ValueError(
                f"Expected 3-D (C, H, W) or 4-D (B, C, H, W) tensor,"
                f" got {tuple(image_input.shape)}."
            )
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @torch.inference_mode()
    def embed_image(self, image_input: str | Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Extracts an L2-normalized 1-D embedding vector of shape ``(d,)`` for one image."""
        self.model.eval()
        tensor_img = self._prepare_tensor(image_input).to(self.device)
        embedding = F.normalize(self.model(tensor_img), p=2, dim=-1)
        return embedding.squeeze(0)

    @torch.inference_mode()
    def _embed_paths(self, image_paths: Sequence[Path], batch_size: int = 64) -> torch.Tensor:
        """Extracts L2-normalized embeddings of shape ``(N, d)`` in mini-batches."""
        if not image_paths:
            raise ValueError("No image paths provided for batch embedding extraction.")

        self.model.eval()
        batches: list[torch.Tensor] = []
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start : start + batch_size]
            tensors = []
            for path in chunk:
                with Image.open(path) as img:
                    tensors.append(self.transform(img.convert("RGB")))
            batch_tensor = torch.stack(tensors, dim=0).to(self.device)
            emb = F.normalize(self.model(batch_tensor), p=2, dim=-1)
            batches.append(emb)

        return torch.cat(batches, dim=0)

    def enroll_person(
        self,
        identity: str,
        image_inputs: Sequence[str | Path | Image.Image | torch.Tensor],
    ) -> None:
        """Enrolls or updates one staff member using one or more reference images."""
        if not image_inputs:
            raise ValueError(f"No reference images provided for identity '{identity}'.")

        embeddings = [self.embed_image(img) for img in image_inputs]
        stacked = torch.stack(embeddings, dim=0)  # (N, d)

        # Average embeddings across reference photos and re-normalize onto unit hypersphere
        mean_emb = torch.mean(stacked, dim=0, keepdim=True)
        norm_emb = F.normalize(mean_emb, p=2, dim=1).squeeze(0)

        if identity in self.gallery_identities:
            idx = self.gallery_identities.index(identity)
            assert self.gallery_embeddings is not None
            self.gallery_embeddings[idx] = norm_emb
        elif self.gallery_embeddings is None:
            self.gallery_embeddings = norm_emb.unsqueeze(0)
            self.gallery_identities.append(identity)
        else:
            self.gallery_embeddings = torch.cat(
                [self.gallery_embeddings, norm_emb.unsqueeze(0)], dim=0
            )
            self.gallery_identities.append(identity)

    def enroll_gallery(self, gallery_dir: str | Path) -> int:
        """Scans a gallery directory and enrolls all reference photos as identity centroids.

        Supports both flat folders (``IdentityName_0001.jpg``) and nested subdirectories
        (``IdentityName/*.jpg``).

        Returns:
            Total number of distinct identities enrolled from ``gallery_dir``.

        Raises:
            FileNotFoundError: If ``gallery_dir`` does not exist.
            ValueError: If no supported image files are found inside ``gallery_dir``.
        """
        root = Path(gallery_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Gallery directory does not exist: {root}")

        identity_to_files: dict[str, list[Path]] = {}

        for file_path in sorted(root.glob("**/*")):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS:
                if file_path.parent == root:
                    # Flat directory: extract identity from filename
                    # (e.g., Colin_Powell_001.jpg)
                    identity = "_".join(file_path.stem.split("_")[:-1]) or file_path.stem
                else:
                    # Nested directory: parent folder name is the identity
                    identity = file_path.parent.name

                identity_to_files.setdefault(identity, []).append(file_path)

        if not identity_to_files:
            raise ValueError(f"No valid images ({sorted(IMAGE_EXTS)}) found in {root}.")

        for identity, files in identity_to_files.items():
            self.enroll_person(identity, files)

        return len(identity_to_files)

    @torch.inference_mode()
    def verify(
        self,
        query_input: str | Path | Image.Image | torch.Tensor,
        threshold: float | None = None,
    ) -> GateDecision:
        """Evaluates whether an incoming query face matches an enrolled identity."""
        if self.gallery_embeddings is None or len(self.gallery_identities) == 0:
            raise RuntimeError("Cannot verify query: Gallery database is empty.")

        active_threshold = self.threshold if threshold is None else float(threshold)
        query_emb = self.embed_image(query_input)  # (d,)

        # Squared Euclidean distance to all enrolled gallery centroids: ||u - v||^2
        diffs = self.gallery_embeddings - query_emb.unsqueeze(0)  # (G, d)
        squared_distances = torch.sum(diffs**2, dim=1)  # (G,)

        min_dist_idx = int(torch.argmin(squared_distances).item())
        min_dist = float(squared_distances[min_dist_idx].item())

        # For unit vectors: ||u - v||^2 = 2 - 2 * cos(u, v) -> cos = 1 - 0.5 * dist^2
        cosine_sim = float(max(-1.0, min(1.0, 1.0 - 0.5 * min_dist)))

        access_granted = min_dist <= active_threshold
        matched_identity = self.gallery_identities[min_dist_idx] if access_granted else None

        return GateDecision(
            access_granted=access_granted,
            matched_identity=matched_identity,
            min_distance=min_dist,
            threshold=active_threshold,
            similarity_score=cosine_sim,
        )

    def extract_probe_distances(
        self, probes_dir: str | Path, batch_size: int = 64
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes minimum gallery distances for all authorized and impostor probes in batches.

        Returns:
            Tuple ``(auth_min_distances, impostor_min_distances)`` of 1-D ``float64`` arrays.
        """
        if self.gallery_embeddings is None or len(self.gallery_identities) == 0:
            raise RuntimeError("Cannot evaluate probes: Gallery database is empty.")

        root = Path(probes_dir)
        auth_dir = root / "authorized"
        impostor_dir = root / "impostors"
        if not auth_dir.is_dir() or not impostor_dir.is_dir():
            raise FileNotFoundError(
                f"Expected 'authorized/' and 'impostors/' subdirectories inside {root}."
            )

        auth_files = sorted(
            [f for f in auth_dir.glob("**/*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        )
        impostor_files = sorted(
            [
                f
                for f in impostor_dir.glob("**/*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ]
        )

        def _min_dists(files: list[Path]) -> np.ndarray:
            if not files:
                return np.empty((0,), dtype=np.float64)
            embs = self._embed_paths(files, batch_size=batch_size)  # (N, d)
            # Squared Euclidean distance matrix (N, G): 2 - 2 * (embs @ gallery.T)
            sq_dists = torch.clamp(2.0 - 2.0 * (embs @ self.gallery_embeddings.T), min=0.0)  # type: ignore[union-attr]
            return sq_dists.min(dim=1).values.cpu().numpy().astype(np.float64)

        return _min_dists(auth_files), _min_dists(impostor_files)

    @staticmethod
    def _score_distances(
        auth_min_dists: np.ndarray,
        impostor_min_dists: np.ndarray,
        threshold: float,
    ) -> GateEvaluationResult:
        """Computes GateEvaluationResult at a given threshold from precomputed min distances."""
        total_auth = int(auth_min_dists.size)
        total_imp = int(impostor_min_dists.size)
        total_attempts = total_auth + total_imp

        true_accepts = int(np.sum(auth_min_dists <= threshold))
        false_rejects = total_auth - true_accepts
        false_accepts = int(np.sum(impostor_min_dists <= threshold))
        true_rejects = total_imp - false_accepts

        tar = true_accepts / max(1, total_auth)
        frr = false_rejects / max(1, total_auth)
        far = false_accepts / max(1, total_imp)
        acc = (true_accepts + true_rejects) / max(1, total_attempts)

        return GateEvaluationResult(
            threshold=float(threshold),
            total_authorized=total_auth,
            total_impostors=total_imp,
            true_accepts=true_accepts,
            false_rejects=false_rejects,
            true_rejects=true_rejects,
            false_accepts=false_accepts,
            tar=float(tar),
            far=float(far),
            frr=float(frr),
            accuracy=float(acc),
        )

    def evaluate_probes(
        self,
        probes_dir: str | Path,
        threshold: float | None = None,
        batch_size: int = 64,
    ) -> GateEvaluationResult:
        """Evaluates gate decisions on authorized staff probes and impostor probes."""
        active_threshold = self.threshold if threshold is None else float(threshold)
        auth_dists, imp_dists = self.extract_probe_distances(probes_dir, batch_size=batch_size)
        return self._score_distances(auth_dists, imp_dists, active_threshold)

    def sweep_thresholds(
        self,
        probes_dir: str | Path,
        thresholds: Sequence[float] | np.ndarray | None = None,
        batch_size: int = 64,
    ) -> list[GateEvaluationResult]:
        """Extracts probe embeddings once and evaluates all thresholds in vectorized NumPy."""
        auth_dists, imp_dists = self.extract_probe_distances(probes_dir, batch_size=batch_size)
        if thresholds is None:
            all_dists = np.concatenate([auth_dists, imp_dists])
            d_min = float(all_dists.min()) if all_dists.size > 0 else 0.0
            d_max = float(all_dists.max()) if all_dists.size > 0 else 4.0
            thresholds = np.linspace(d_min, d_max, num=50)
        return [self._score_distances(auth_dists, imp_dists, float(tau)) for tau in thresholds]

    def save_database(self, file_path: str | Path) -> None:
        """Saves enrolled gallery embeddings and identity list to a checkpoint."""
        if self.gallery_embeddings is None or len(self.gallery_identities) == 0:
            raise RuntimeError("Cannot save database: Gallery is empty.")

        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": self.gallery_embeddings.cpu(),
                "identities": self.gallery_identities,
                "threshold": self.threshold,
            },
            out_path,
        )

    def load_database(self, file_path: str | Path) -> None:
        """Loads enrolled gallery database from a checkpoint file."""
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=True)
        self.gallery_embeddings = checkpoint["embeddings"].to(self.device)
        self.gallery_identities = list(checkpoint["identities"])
        self.threshold = float(checkpoint.get("threshold", self.threshold))
