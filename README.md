# Heart Disease Classification

> Binary classification of cardiovascular disease risk using CDC BRFSS 2020 survey data.  
> **Hochschule Pforzheim · BIS3012 · Prof. Dr. Dustin van der Haar · 2026**

🌐 **Live website:** [saffadek.github.io/heart-disease-classification]([https://saffadek.github.io/heart-disease-classification](https://saffadek.github.io/heart-disease-classification))

---

## Team

| Name | Role |
|---|---|
| Karim | Data Engineer — preprocessing, sampling, encoding, repo structure |
| Kushul | Code Reviewer — code quality, bug fixes, train.py |
| Tarik | Analyst — evaluation, results, presentation |
| Max | Communicator — report, ethics discussion |

---

## Project Overview

We sampled **2,059 patients** from 319,795 CDC BRFSS 2020 survey responses — preserving the real-world disease rate of 8.6% — and trained three ML classifiers to predict heart disease from 17 self-reported health and lifestyle measurements.

**Task:** Binary Classification — `0` = No Heart Disease, `1` = Heart Disease  
**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2020  
**Best model:** Random Forest — 82.3% test accuracy, AUC-ROC 0.855  
**Imbalance:** Handled via `class_weight='balanced'` on all models  

---

## Dataset

| | Value |
|---|---|
| Source | CDC BRFSS 2020 (Kamilpytlak, 2021 — Kaggle) |
| Full dataset | 319,795 responses |
| Our sample | 2,059 patients |
| Disease rate | 8.6% — real-world CDC rate preserved |
| Features | 17 |
| Missing values | 0 |

**Why we kept the real imbalance:**  
A forced 50/50 split misrepresents reality. We preserved the natural 8.6% disease rate and handled imbalance correctly with `class_weight='balanced'` — the academically standard approach.

---

## Repository Structure

```
heart-disease-classification/
├── notebooks/
│   └── Heart_Disease_FINAL.ipynb    ← main analysis (runs in Google Colab)
├── src/
│   ├── preprocess.py                ← data loading, sampling, encoding
│   ├── train.py                     ← GridSearchCV model training
│   └── evaluate.py                  ← metrics, charts, reports
├── docs/
│   └── index.html                   ← project website (GitHub Pages)
├── Heart_Disease_Report_FINAL.pdf   ← written academic report
└── README.md
```

---

## How to Run

### Google Colab (recommended)
1. Open `notebooks/Heart_Disease_FINAL.ipynb` in Google Colab
2. Run the first cell → click **Choose Files** → select `heart_2020_cleaned.csv`
3. Run all cells top to bottom

### Local
```bash
git clone https://github.com/saffadek/heart-disease-classification
cd heart-disease-classification
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Download heart_2020_cleaned.csv from:
# https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
# Place it in the data/ folder, then:

python src/preprocess.py   # encode and split data
python src/train.py        # train all 3 models
python src/evaluate.py     # generate metrics and charts
```

---

## Results

| Model | Best Params | CV Accuracy | Test Accuracy | Recall (Disease) | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | C=0.01 | 77.5% ±1.3% | 76.5% | **80.0%** | 0.849 |
| SVM (RBF) | C=0.1, γ=0.001 | 86.9% ±2.0% | 84.2% | 54.3% | 0.828 |
| **Random Forest** | **depth=5, n=100, split=10** | **82.6% ±1.6%** | **82.3%** | 74.3% | **0.855 ★** |

All models use `class_weight='balanced'` and GridSearchCV with F1 scoring, 5-fold stratified CV.

### Why SVM's 84% accuracy is misleading
SVM predicts "no disease" for most borderline cases to exploit the 91.4% majority class — giving high accuracy but only 54% disease recall. AUC (0.828) and recall tell the real story.

### Top 5 Features (Random Forest)
| # | Feature | Importance | Meaning |
|---|---|---|---|
| 1 | AgeCategory | 28.7% | Disease risk rises sharply after 50 |
| 2 | GenHealth | 13.4% | Self-reported health captures comorbidity |
| 3 | DiffWalking | 10.0% | Reflects physical deconditioning |
| 4 | PhysicalHealth | 8.6% | More bad days = higher illness burden |
| 5 | Diabetic | 8.2% | Independent cardiovascular risk factor |

---

## Ethics

- **Bias:** US-only dataset with race as a feature — subgroup evaluation required before deployment
- **Self-reported data:** Survey responses introduce recall bias and measurement error
- **False negatives:** Missed diagnoses are dangerous — handled via F1 tuning and class_weight
- **Privacy:** GDPR (EU) and HIPAA (US) apply to any clinical deployment
- **Role:** Decision-support tool only — physician always makes the final call

---

## References

- Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and machine learning.* fairmlbook.org.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- CDC. (2020). *Behavioral Risk Factor Surveillance System.* https://www.cdc.gov/brfss/
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297.
- European Parliament. (2016). *General Data Protection Regulation (GDPR).*
- Kamilpytlak. (2021). *Personal key indicators of heart disease.* Kaggle.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS, 30.*
- Pedregosa, F., et al. (2011). Scikit-learn. *JMLR, 12*, 2825–2830.
- World Health Organization. (2023). *Cardiovascular diseases fact sheet.* https://www.who.int
