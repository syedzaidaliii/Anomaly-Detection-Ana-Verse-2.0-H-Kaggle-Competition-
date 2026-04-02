# Anomaly Detection — Ana-Verse 2.0-H (Kaggle Competition)

A binary classification pipeline that identifies anomalies in multivariate time-series sensor data. The solution combines LightGBM and XGBoost in a weighted ensemble with optimized decision thresholds, achieving an **F1 score of ~0.82** on the held-out validation set.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Key Design Decisions](#key-design-decisions)

---

## Problem Statement

Given time-stamped sensor readings (`X1`–`X5`) collected over several years, predict whether each observation is **normal (0)** or an **anomaly (1)**. The challenge is heavily imbalanced — anomalies make up less than 1% of the training data — so standard accuracy is meaningless here; the competition metric is **F1 score on the positive class**.

---

## Dataset

| Split | Rows | Columns |
|-------|------|---------|
| Train | 1,639,424 | Date, X1–X5, target |
| Test  | 409,856   | ID, Date, X1–X5 |

**Source:** [Kaggle — Ana-Verse 2.0-H](https://www.kaggle.com/competitions/ana-verse-2-0-h)

The raw files are in Parquet format. You need to download them from Kaggle and place them at:

```
/kaggle/input/ana-verse-2-0-h/train.parquet
/kaggle/input/ana-verse-2-0-h/test.parquet
/kaggle/input/ana-verse-2-0-h/sample_submission.parquet
```

**Class imbalance in training set:**
- Normal (0): 1,300,309 samples
- Anomaly (1): 11,230 samples
- Ratio ≈ 116 : 1

---

## Approach

### 1. Exploratory Data Analysis
- Class distribution bar chart to visualise the imbalance
- Box plots of each sensor signal stratified by target class
- Pearson correlation heatmap across all five sensors

### 2. Feature Engineering (134 features total)

Starting from the five raw sensor readings and the timestamp:

**Date decomposition**
- Day of month, weekday, month

**Cross-sensor aggregates** (row-wise)
- Sum, mean, standard deviation, max, min, range

**Per-sensor temporal features**
- Lag features: t-1, t-2, t-3
- First-order differences: diff1, diff2, diff3
- Rolling statistics (windows 3, 5, 10, 20): mean, std, min, max
- Exponential moving averages: EMA-5, EMA-10

All features are computed after sorting by date, so there is no look-ahead leakage.

### 3. Model Training

**LightGBM**
```
n_estimators    = 3000
learning_rate   = 0.02
num_leaves      = 64
subsample       = 0.8
colsample_bytree= 0.8
min_child_samples = 30
reg_lambda      = 2.0
scale_pos_weight = ~115.8
```

**XGBoost**
```
n_estimators    = 2000
max_depth       = 7
learning_rate   = 0.03
subsample       = 0.8
colsample_bytree= 0.8
reg_lambda      = 2.0
scale_pos_weight = ~115.8
tree_method     = hist
```

`scale_pos_weight` is set to `(# negatives) / (# positives)` to compensate for the class imbalance.

### 4. Ensemble + Threshold Optimisation

The final prediction is a weighted average of both models' probability outputs:

```
prob = 0.6 × lgbm_prob + 0.4 × xgb_prob
```

Weights and the classification threshold are jointly swept on the validation set and chosen by maximising F1 on the positive class. The optimal threshold found was **≈ 0.864**.

---

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 99.71% |
| F1 Score (anomaly class) | **0.8215** |
| Precision (anomaly class) | 0.87 |
| Recall (anomaly class) | 0.78 |

Confusion matrix (validation set, 327,885 samples):

```
              Predicted 0   Predicted 1
Actual 0        324,739          338
Actual 1            615        2,193
```

---

## Project Structure

```
.
├── syed-zaid-ali.ipynb     # Main notebook (EDA → feature engineering → training → submission)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

The notebook is self-contained and runs end-to-end when the Kaggle dataset is available at the expected paths.

---

## Setup & Installation

**Python version:** 3.12 (tested on 3.12.12)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

> If you are running on Kaggle, all packages are already available in the standard GPU kernel image — no installation needed.

---

## How to Run

1. **On Kaggle (recommended)**
   - Upload the notebook to Kaggle, attach the *ana-verse-2-0-h* dataset, and enable GPU acceleration (NVIDIA Tesla T4).
   - Click *Run All*. Total wall-clock time is roughly 40 minutes.

2. **Locally**
   - Download the dataset from the competition page and adjust the file paths in cells 2 and 20 accordingly.
   - Make sure you have a GPU with CUDA available for reasonable training times, or be prepared to wait longer on CPU.

---

## Key Design Decisions

**Why sort by date before rolling features?**
Rolling windows and lag features must be computed on chronologically ordered data. Sorting by date first ensures the temporal context captured by these features is meaningful and leak-free.

**Why `scale_pos_weight` rather than oversampling?**
With ~1.6 million training rows, resampling would dramatically increase memory usage. Setting `scale_pos_weight` achieves a similar effect at essentially zero cost.

**Why threshold tuning?**
Both models are calibrated to output probabilities, but the default 0.5 cutoff is suboptimal when precision and recall need to be traded off differently. Sweeping thresholds from 0.05 to 0.99 in 300 steps and picking the F1-maximising value consistently adds a few points of F1 over the default.

**Why not use `early_stopping_rounds`?**
A fixed number of estimators was used intentionally so that the validation set is kept clean for threshold optimisation — mixing early stopping with threshold tuning on the same fold can introduce subtle bias.
