"""
preprocess.py
=============
Heart Disease Classification — CDC BRFSS 2020
Hochschule Pforzheim · BIS3012 · 2026

Loads the CDC BRFSS 2020 dataset, samples 2,059 patients at the real-world
disease rate (8.6%), encodes all features, and splits into train/test sets.

Usage:
    python src/preprocess.py

Output:
    data/X_train.csv, data/X_test.csv,
    data/y_train.csv, data/y_test.csv
"""

import pandas as pd
import numpy as np
import os

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH    = "data/heart_2020_cleaned.csv"
OUTPUT_DIR   = "data"
RANDOM_STATE = 42
N_DISEASE    = 177    # real CDC disease rate ~8.6%
N_NO_DISEASE = 1882
TEST_SIZE    = 0.2

# ── Encoding maps ─────────────────────────────────────────────────────────────
AGE_MAP = {
    '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3,
    '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7,
    '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11,
    '80 or older': 12
}
GEN_HEALTH_MAP = {
    'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4
}
DIABETIC_MAP = {
    'No': 0,
    'No, borderline diabetes': 1,
    'Yes (during pregnancy)': 1,
    'Yes': 2
}
BINARY_COLS = [
    'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking',
    'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer'
]


def load_and_sample(path: str) -> pd.DataFrame:
    """Load full CDC dataset and sample 2,059 patients at real disease rate."""
    print(f"Loading dataset from: {path}")
    df_full = pd.read_csv(path)
    print(f"  Full dataset: {df_full.shape[0]:,} patients, {df_full.shape[1]} features")
    print(f"  Disease rate: {(df_full['HeartDisease']=='Yes').mean()*100:.1f}%")

    # Sample at real CDC disease rate
    disease    = df_full[df_full['HeartDisease'] == 'Yes'].sample(N_DISEASE,    random_state=RANDOM_STATE)
    no_disease = df_full[df_full['HeartDisease'] == 'No'].sample(N_NO_DISEASE,  random_state=RANDOM_STATE)
    df = (pd.concat([disease, no_disease])
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True))

    print(f"  Sample: {len(df)} patients")
    print(f"    Disease:    {(df['HeartDisease']=='Yes').sum()} ({(df['HeartDisease']=='Yes').mean()*100:.1f}%)")
    print(f"    No Disease: {(df['HeartDisease']=='No').sum()} ({(df['HeartDisease']=='No').mean()*100:.1f}%)")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical features into numeric form.

    Encoding strategy:
    - Binary Yes/No columns  → 0 / 1
    - Sex                    → 0 (Female) / 1 (Male)
    - AgeCategory            → ordinal 0–12
    - GenHealth              → ordinal 0–4
    - Diabetic               → ordinal 0–2
    - Race                   → one-hot (drop first to avoid multicollinearity)
    - HeartDisease (target)  → 0 / 1
    """
    df_enc = df.copy()

    # Binary columns
    for col in BINARY_COLS:
        df_enc[col] = (df_enc[col] == 'Yes').astype(int)

    df_enc['Sex']         = (df_enc['Sex'] == 'Male').astype(int)
    df_enc['AgeCategory'] = df_enc['AgeCategory'].map(AGE_MAP)
    df_enc['GenHealth']   = df_enc['GenHealth'].map(GEN_HEALTH_MAP)
    df_enc['Diabetic']    = df_enc['Diabetic'].map(DIABETIC_MAP)

    # One-hot encode Race (drop first to avoid multicollinearity)
    race_dummies = pd.get_dummies(df_enc['Race'], prefix='Race', drop_first=True).astype(int)
    df_enc = pd.concat([df_enc.drop('Race', axis=1), race_dummies], axis=1)

    # Target
    df_enc['HeartDisease'] = (df_enc['HeartDisease'] == 'Yes').astype(int)

    missing = df_enc.isnull().sum().sum()
    print(f"  Encoded shape: {df_enc.shape}, Missing values: {missing}")
    assert missing == 0, f"Encoding produced {missing} missing values — check encoding maps."

    return df_enc


def split_and_save(df_enc: pd.DataFrame, output_dir: str) -> None:
    """Split into train/test sets and save to CSV."""
    from sklearn.model_selection import train_test_split

    X = df_enc.drop('HeartDisease', axis=1)
    y = df_enc['HeartDisease']

    # Stratified split preserves the 8.6% disease rate in both subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"  Train: {len(X_train)} patients (disease: {y_train.sum()})")
    print(f"  Test:  {len(X_test)} patients (disease: {y_test.sum()})")

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv",  index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv",  index=False)
    print(f"  Saved to {output_dir}/")


def main():
    print("=" * 55)
    print("PREPROCESSING — CDC BRFSS 2020 Heart Disease Dataset")
    print("=" * 55)

    print("\n[1/3] Loading and sampling...")
    df = load_and_sample(DATA_PATH)

    print("\n[2/3] Encoding features...")
    df_enc = encode_features(df)

    print("\n[3/3] Splitting and saving...")
    split_and_save(df_enc, OUTPUT_DIR)

    print("\nPreprocessing complete.")
    print(f"Features: {df_enc.shape[1] - 1}")
    print(f"Run 'python src/train.py' next.")


if __name__ == "__main__":
    main()
