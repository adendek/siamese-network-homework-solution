# DeepGate-FaceRec — Dataset Documentation

This directory contains the curated and split face dataset used for the **DeepGate-FaceRec** homework assignment. It is derived from the **Labeled Faces in the Wild (LFW)** benchmark (deep-funneled alignment version).

---

## 1. Dataset Overview & Curation Logic

In biometric metric learning and face verification, standard random image splitting causes **severe data leakage**. If images of the same person are placed in both training and test sets, the model could simply memorize subject-specific features (such as hair color or clothing) rather than learning a generalizable metric embedding space $\mathbb{R}^d$.

To prevent this, the dataset is strictly **identity-disjoint**:
$$\text{Identities}(\text{Train}) \cap \text{Identities}(\text{Val}) \cap \text{Identities}(\text{Test}) = \emptyset$$

### Filtering Criteria
The raw LFW corpus contains 5,749 identities, but the majority have only a single image. We filtered the dataset using the following rules:
- **Minimum images per identity ($> 4$):** Ensures that every identity can form valid anchor-positive $(a, p)$ pairs during $P \times K$ batch construction.
- **Maximum images per identity ($< 100$):** Excludes dominant public figures (e.g., George W. Bush with 500+ photos) to avoid severe class imbalance and batch dominance during triplet mining.

---

## 2. Directory Layout

The prepared dataset is organized under `data/split/` as follows:

```text
data/
├── README.md                     # This file
├── for_students/
└── split/
    ├── train_identities.csv      # List of identities assigned to training
    ├── val_identities.csv        # List of identities assigned to validation
    ├── test_identities.csv       # List of identities assigned to testing
    │
    ├── train/                    # ~70% of identities (Metric representation learning)
    │   ├── Colin_Powell/
    │   │   ├── Colin_Powell_0001.jpg
    │   │   └── Colin_Powell_0002.jpg
    │   └── Tony_Blair/
    │       └── ...
    │
    ├── val/                      # ~15% of identities (Open-set validation monitoring)
    │   ├── Gerhard_Schroeder/
    │   └── ...
    │
    └── test/                     # ~15% of identities (Simulated Smart Access Gate)
        ├── gate_gallery/         # Enrolled reference database (1 reference photo per staff member)
        │   ├── Staff_PersonA_0001.jpg
        │   └── Staff_PersonB_0001.jpg
        │
        └── gate_probes/          # Access attempts at the smart gate
            ├── authorized/       # Unseen query photos of enrolled staff (Ground Truth: UNLOCK)
            │   ├── Staff_PersonA_0002.jpg
            │   └── Staff_PersonB_0002.jpg
            └── impostors/        # Query photos of un-enrolled strangers (Ground Truth: DENY)
                ├── Impostor_PersonX_0001.jpg
                └── Impostor_PersonY_0001.jpg
