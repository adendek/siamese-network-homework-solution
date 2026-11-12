# DeepGate-FaceRec

**Deep Neural Networks — Homework Assignment: Face Recognition via Deep Metric & Embedding Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-8A2BE2.svg)](https://docs.astral.sh/uv/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

> **Assignment Objective:** Move beyond traditional softmax classification. In this homework, you will train a Siamese/Triplet architecture that learns a metric embedding space $\mathbb{R}^d$ where faces of the same identity cluster closely together, and different identities are pushed apart by a margin $\alpha$. You will then use these embeddings to simulate an automated **Smart Access Gate**.

---

## 1. Homework Overview & Learning Milestones

Instead of training a neural network to predict a closed-set categorical label (which fails whenever an unseen person arrives), modern biometric systems train an **embedding function** $f_\theta(\mathbf{x}) \in \mathbb{R}^d$.

To guide your implementation, the homework is organized into six progressive milestones:

| Milestone | Topic | Description | Deliverables |
|---|---|---|---|
| **Part 1** | **Embedding Architecture** | Adapt a pretrained vision backbone (ResNet/MobileNet) with an embedding projection head and $L_2$ hypersphere normalization. | `src/models.py` |
| **Part 2** | **Identity Batch Sampling** | Implement a $P \times K$ identity sampler (sample $P$ subjects, each with $K$ distinct face images) to enable informative in-batch comparisons. | `src/dataset.py` |
| **Part 3** | **Triplet Loss from Scratch** | Implement vectorised Triplet Margin Loss: $\mathcal{L}(a, p, n) = \max\left(0, \|\mathbf{z}_a - \mathbf{z}_p\|_2^2 - \|\mathbf{z}_a - \mathbf{z}_n\|_2^2 + \alpha\right)$ without calling `torch.nn.TripletMarginLoss`. | `src/loss.py` |
| **Part 4** | **Online Triplet Mining** | Implement batch-hard and semi-hard negative mining strategies using pairwise distance matrices. | `src/mining.py` |
| **Part 5** | **Smart Gate Simulation** | Enroll staff identities, evaluate True/False Acceptance Rates (TAR/FAR), and simulate real-time gate pass/deny decisions on open-set query faces. | `src/smart_gate.py`, `02_smart_gate_demo.ipynb` |
| **Part 6** | **Scientific Report** | Write a conference-style report documenting mathematical formulations, mining comparisons, metric curves, and threshold trade-offs. | `docs/main.pdf` |

---

## 2. Repository Layout

The project follows a modular, production-ready structure:

```text
deepgate-facerec/
├── .github/
│   └── workflows/
│       ├── ci.yml               # Automated linting, code formatting, and type checking
│       └── tests.yml            # Unit tests and grading checks
├── .pre-commit-config.yaml      # Git hooks for local code hygiene
├── pyproject.toml               # Single source of truth for build, dependencies & tooling
├── README.md                    # Assignment instructions and reference guide
├── configs/
│   └── config.yaml              # Hyperparameters (margin, embedding_dim, lr, P, K, epochs)
├── data/
│   ├── download_data.py         # Automated dataset download and extraction script
│   └── raw/                     # Face dataset directory (ignored by git)
├── docs/                        # LaTeX report template and instructions
│   ├── README.md                # Report guide and Overleaf setup instructions
│   ├── deepgate_report.sty      # NeurIPS-adapted LaTeX style package
│   ├── main.tex                 # Report master file
│   ├── references.bib           # Bibliography references
│   └── figures/                 # Report plots (ROC curves, UMAP embeddings)
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Identity-grouped dataset and PK batch sampler
│   ├── models.py                # Pretrained backbone + projection head
│   ├── loss.py                  # [TODO: Student Exercise] Custom Triplet Loss
│   ├── mining.py                # [TODO: Student Exercise] Online hard/semi-hard mining
│   ├── trainer.py               # Metric learning training loop & distance logging
│   ├── evaluate.py              # Verification metrics (ROC curve, AUC, EER, distance histograms)
│   └── smart_gate.py            # [TODO: Student Exercise] Enrollment and threshold verification
├── notebooks/
│   ├── 01_explore_embeddings.ipynb  # t-SNE / UMAP visualization (pre vs. post training)
│   └── 02_smart_gate_demo.ipynb     # Interactive simulation of gate access decisions
└── tests/
    ├── test_loss.py             # Grader tests for TripletLoss accuracy and gradients
    ├── test_mining.py           # Grader tests for triplet mining logic and batch masking
    └── test_models.py           # Output shape and L2 normalization tests
```

---

## 3. Engineering Stack & Tooling

To ensure scientific rigor and clean software engineering habits, this project uses:

### A. Environment & Package Management: [`uv`](https://docs.astral.sh/uv/)
* **Deterministic, Blazing-Fast Setup**: Resolves and installs PyTorch and dependencies in seconds.
* **Hermetic Execution**: Run scripts and tools via `uv run <command>` without worrying about global environment pollution.

### B. Fast Linting & Formatting: [`ruff`](https://docs.astral.sh/ruff/)
* Replaces Black, Flake8, and isort with sub-second execution.
* Google-style docstrings and PEP 8 standards are enforced. Line length is set to **88 characters**.

### C. Static Type Checking: [`mypy`](https://mypy-lang.org/)
* Enforces type annotations across model inputs, loss functions, and tensor shapes to prevent dimensional mismatches early.

### D. Automated Quality Gate: [`pre-commit`](https://pre-commit.com/)
* Automatically validates formatting, type contracts, and tests before any commit is finalized.

---

## 4. Getting Started

### Prerequisites
* **Python**: `3.11` or newer
* **uv**: Install via Astral's official script:
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### Local Setup in 4 Commands

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-course/DeepGate-FaceRec.git
   cd DeepGate-FaceRec
   ```

2. **Sync the virtual environment and all dependencies**:
   ```bash
   uv sync --all-extras --group dev
   ```

3. **Install Git pre-commit hooks**:
   ```bash
   uv run pre-commit install --install-hooks
   ```

This is optional for now.

4. **Download and prepare the face dataset**:
   ```bash
   uv run python data/download_data.py
   ```

---

## 5. Development & Testing Command Reference

All commands must be executed using `uv run` to guarantee they run inside the isolated environment:

| Task | Command | Description |
|---|---|---|
| **Run All Unit Tests** | `uv run pytest` | Runs all unit tests with verbose test reporting |
| **Test Specific Module** | `uv run pytest tests/test_loss.py` | Runs tests only for your Triplet Loss implementation |
| **Check Test Coverage** | `uv run pytest --cov=src` | Computes test coverage across your implementation |
| **Lint Code** | `uv run ruff check .` | Checks for linting issues and code style bugs |
| **Auto-fix Lint** | `uv run ruff check . --fix` | Automatically fixes safe linting errors |
| **Format Code** | `uv run ruff format .` | Formats all files according to project style |
| **Type Check** | `uv run mypy src/` | Runs strict static type validation |
| **Run Pre-Commit** | `uv run pre-commit run --all-files` | Manually triggers all pre-commit hooks |
| **Start Notebook** | `uv run jupyter lab` | Launches JupyterLab to run the visual demos |

---

## 6. Written Report & Overleaf Template

Alongside the working code, you must submit a concise, conference-style scientific report (3–4 pages). The report format is adapted from the NeurIPS conference style.

### Direct Overleaf Template Link
You can open and copy the official course report template directly in your browser:
👉 **[Open DeepGate-FaceRec Report Template on Overleaf](https://www.overleaf.com/project/6ae88d3923251984d11a96a5/share#2edcd42a0d900d6fa592d1db61885d531af4d654656f98ee)**

*(Inside Overleaf, click **Menu** $\rightarrow$ **Copy Project** to create your own editable team copy).*

### Required Report Content
1. **Mathematical Formulations**: State the exact loss formula, distance metric, and how gradients backpropagate through normalized embeddings.
2. **Mining Analysis**: Empirical comparison between Random, Semi-Hard, and Batch-Hard mining (loss convergence, percentage of non-zero loss triplets).
3. **Verification Performance**: ROC curve, AUC, Equal Error Rate (EER), and distance distribution histograms for positive vs. negative pairs.
4. **Smart Gate Decision Trade-offs**: Analysis of False Acceptance Rate (FAR) vs. False Rejection Rate (FRR) as a function of the threshold $\tau$, explaining the security implications of choosing conservative vs. permissive thresholds.

See [`docs/report/README.md`](docs/report/README.md) for full compilation instructions, styling details, and formatting tips.

---

## 7. Submission Guidelines

Your final submission consists of two deliverables:

1. **Codebase**:
   - All tests in `tests/` pass (`uv run pytest`).
   - Clean code hygiene passing Ruff and Mypy (`uv run ruff check .` and `uv run mypy src/`).
   - Push your branch to the course GitHub repository.

2. **Compiled Report (`report.pdf`)**:
   - Compiled PDF from the Overleaf template placed in `docs/report.pdf` (or uploaded to the course submission portal).
   - Ensure all student names, IDs, and figures (ROC curves, UMAP projections) are clearly legible.
```
