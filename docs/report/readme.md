# DeepGate-FaceRec: Report Documentation & LaTeX Template

This directory contains the official LaTeX template for your **DeepGate-FaceRec** homework report. The layout is adapted from the Neural Information Processing Systems (NeurIPS) camera-ready format, configured specifically for academic coursework submissions.

---

## 1. Directory Structure

```text
docs/report/
├── README.md               # This instruction file
├── deepgate_report.sty     # LaTeX package defining typography, margins, and headers
├── main.tex                # Main report document (edit your text here)
├── references.bib          # BibTeX bibliography file
└── figures/                # Directory for plots, diagrams, and architecture figures
    ├── umap_clusters.png   # UMAP embedding separation plot
    └── roc_curve.png       # Receiver Operating Characteristic plot
```

---

## 2. Report Requirements & Guidelines

- **Recommended Length:** 3 to 4 pages (excluding references section).
- **Target Audience:** Write with academic rigor. State exact mathematical formulations, describe training hyperparameters, and support experimental conclusions with empirical metrics.
- **Key Sections to Cover:**
  1. **Introduction & Motivation:** Contrast softmax classification with metric embedding learning; formulate the open-set verification problem.
  2. **Methodology:**
     - Pretrained vision backbone and $L_2$ hypersphere projection head.
     - Identity-aware batch sampling ($P \times K$ batch construction).
     - Vectorized Triplet Margin Loss formulation.
     - Online mining strategies (Random vs. Semi-Hard vs. Batch-Hard).
  3. **Experimental Setup & Verification Metrics:**
     - Dataset split (training identities vs. held-out test identities).
     - Metric evaluation: False Acceptance Rate (FAR), False Rejection Rate (FRR), Equal Error Rate (EER), Area Under ROC Curve (AUC).
     - Qualitative embedding geometry (t-SNE or UMAP visualizations).
  4. **Smart Gate Simulation:**
     - Gallery enrollment protocol and query matching logic.
     - Threshold sensitivity analysis ($\tau$ sweep) and access decision trade-offs.
  5. **Conclusion & Generative AI Declaration:** Summary of findings and statement regarding tools utilized.

---

## 3. Working with Overleaf (Recommended for Easy Setup)

Overleaf provides a browser-based, zero-configuration LaTeX environment. Follow these steps to set up your project.


### Step 1: Package the Template
Zip the contents of the `docs/report` folder:

```bash
# On Linux / macOS
cd docs/report
zip -r deepgate_report_template.zip deepgate_report.sty main.tex references.bib figures/ README.md

# On Windows (PowerShell)
cd docs
Compress-Archive -Path deepgate_report.sty, main.tex, references.bib, figures -DestinationPath deepgate_report_template.zip
```

### Step 2: Import into Overleaf
1. Log in to [Overleaf](https://www.overleaf.com/).
2. In your dashboard, select **New Project** $\rightarrow$ **Upload Project**.
3. Upload `deepgate_report_template.zip`. Overleaf will extract all files and open `main.tex`.

### Step 3: Configure Project Settings
In the top-left menu inside Overleaf (**Menu** icon):
- **Compiler:** Set to `pdfLaTeX` (default).
- **TeX Live version:** Select `2024` or latest.
- **Main document:** Ensure it is set to `main.tex`.

### Step 4: Submission Mode (Anonymous vs. Final)
In `main.tex`, you can toggle between anonymous grading mode and final named submission:

```latex
\usepackage[final]{deepgate_report}
```

### Step 5: Exporting Final PDF
Once your team finishes drafting:
1. Click **Recompile** to ensure all citations and figure labels are resolved.
2. Click the **Download PDF** icon next to the Recompile button.
3. Verify that the generated PDF does not contain unrendered citations (`[?]`) or missing references.

---

## 4. Local Compilation (Alternative)

If you prefer editing locally with VS Code (using the *LaTeX Workshop* extension) or compiling via the command line:

### Prerequisites
A complete TeX distribution is required:
- **macOS:** MacTeX (`brew install --cask mactex`)
- **Ubuntu/Debian:** TeX Live (`sudo apt install texlive-latex-extra texlive-science texlive-fonts-recommended`)
- **Windows:** MiKTeX or TeX Live

### Compilation Commands

```bash
cd docs

# Standard compilation sequence to resolve cross-references and bibliography
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
