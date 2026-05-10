# AMOSUM-ML: Adaptive Propagator Selection via Machine Learning

> **Automatically selecting the best AMOSUM propagator configuration on a per-instance basis using Machine Learning.**

---

## Overview

Answer Set Programming (ASP) is a powerful declarative paradigm for knowledge representation and automated reasoning, where solutions are encoded as stable models of logic programs. A recurring pattern in real-world ASP problems is the co-occurrence of **SUM constraints** — which assert that a weighted sum of literals meets a given threshold — and **At-Most-One (AMO) constraints**, which restrict the simultaneous truth of elements within a set.

The **AMOSUM propagator** was introduced to jointly handle these two constraint types, achieving significant performance gains over independent treatment. Building on this, several enhanced configurations of the AMOSUM propagator have been proposed — yet no single configuration dominates across all problem instances.

This project addresses that gap with a **Machine Learning-based approach** that automatically selects the most effective propagator configuration for each problem instance, using semantically meaningful features extracted from the interaction between SUM and AMO constraints.

**Key results:** ML-based selectors substantially outperform any single static configuration in both number of solved instances and average runtime.

---

## How It Works

```
Problem Instance
      │
      ▼
Feature Extraction  ──►  SUM × AMO interaction features
      │
      ▼
ML Selector  ──►  Best AMOSUM configuration
      │
      ▼
clingo + AMOSUM  ──►  Solution
```

1. **Feature extraction** — Semantic features capturing the structural relationship between SUM and AMO constraints are extracted from each instance.
2. **ML-based selection** — A trained classifier maps instance features to the predicted best propagator configuration.
3. **Solving** — The selected configuration is passed to `clingo` with the AMOSUM propagator.

---

## Benchmarks

The training dataset is built from three combinatorial benchmark families, providing diverse coverage of SUM+AMO interaction patterns.

---

## Requirements

### AMOSUM Solver

| Dependency | Version |
|------------|---------|
| `clingo`   | 5.8.0   |
| `g++`      | any recent |
| `Python`   | 3.10    |
| `make`     | any     |

### ML Components

> *(List your ML dependencies here, e.g. `scikit-learn`, `pandas`, `numpy`, etc.)*

---

## Installation

### 1. Install the AMOSUM propagator

```bash
pip install AMOSUM/.
bash AMOSUM/install.sh
```

### 2. Install ML dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Construction

Building the full training dataset involves two steps:

### Step 1 — Extract MEASP features

```bash
python dataset_building_measp_split.py
```

This script extracts semantic features from each instance, capturing the interaction structure between SUM and AMO constraints.

### Step 2 — Build the complete dataset

```bash
python split_datasets.py
```

This merges the MEASP features with AMO-specific features and produces the final dataset splits used for training and evaluation.

---

## Usage

> *(Add instructions here for training the ML selector and running inference on new instances.)*

```bash
# Example: train the selector
python train_selector.py --dataset data/full_dataset.csv

# Example: predict configuration for a new instance
python predict.py --instance path/to/instance.lp
```

---

## Project Structure

```
.
├── AMOSUM/                  # AMOSUM propagator source and installer
├── data/                    # Datasets and benchmark instances
├── features/                # Feature extraction scripts
│   └── dataset_building_measp_split.py
├── splits/
│   └── split_datasets.py
├── models/                  # Trained ML selectors
├── train_selector.py        # Training pipeline
├── predict.py               # Inference script
└── README.md
```

---

## Citation

> *(Add your paper reference here once published.)*

```bibtex
@article{yourname2025amosumml,
  title   = {ML-Based Configuration Selection for the AMOSUM Propagator},
  author  = {Your Name},
  year    = {2025}
}
```

---

## License

> *(Specify your license here.)*