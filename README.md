# Fake-News Detection with a Hierarchical Attention Network

  

## Goal

Train and evaluate a Hierarchical Attention Network (HAN) that classifies news articles as **fake** or **true**.

  

## Contents

* Python implementation of the HAN architecture (word- and sentence-level GRUs with attention)
* Data preprocessing and training utilities
* Jupyter pipeline to reproduce all results
* Model card in LaTeX with the final hyper-parameters and metrics

  

## Quick Start

  

### 1 · Set up a virtual environment

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

  

### 2 · Run the training / validation pipeline

Open `pipeline.ipynb`, adjust the control flags at the top
(`ENABLE_TRAINING`, `ENABLE_KFOLD_CV`, `ENABLE_BOOTSTRAP_VALIDATION`, etc.) and run all cells.

  

Outputs

* `best_han_model.pth` — best checkpoint
* `plots/` — ROC, precision–recall, k-fold, bootstrap, confusion matrices
* printed metrics for training, validation and held-out test splits

  

### 3 · Evaluate on another dataset

Open `external_datasets_test.ipynb`, set the CSV path and label mapping, run all cells.

The notebook loads the saved vocabulary and checkpoint, then prints accuracy and a classification report.

  

## Experiments Conducted

* Context window: 20 × 50 (paper) vs 30 × 60 (default) sentences × tokens
* Vocabulary size 30k–75k, min-frequency 2-3
* Dropout 0.1–0.3, label smoothing 0.05
* Encoder depth 1–3 GRU layers
* Title inclusion / exclusion
* Token-level augmentation (random deletions and swaps)

  

The hyper-parameters giving the best external-dataset performance are recorded in `model_card/model_card.tex`.