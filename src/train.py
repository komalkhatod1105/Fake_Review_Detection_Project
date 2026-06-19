#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from clean_text import batch_clean
from features import extract_numeric_features


# -------------------- DATA LOADING --------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic validation
    if "Review Text" not in df.columns or "Rating" not in df.columns:
        raise ValueError("CSV must contain 'Review Text' and 'Rating' columns")

    df["text"] = df["Review Text"].fillna("").astype(str)

    # FIXED: better rating parsing
    def make_label(x):
        try:
            x = str(x).lower()
            if "1" in x or "2" in x:
                return "FAKE"
            else:
                return "REAL"
        except:
            return "REAL"

    df["label"] = df["Rating"].apply(make_label)

    label_map = {"FAKE": 1, "REAL": 0}
    df["y"] = df["label"].map(label_map)

    df["text_clean"] = batch_clean(df["text"])

    return df[["text", "text_clean", "y"]]


# -------------------- NUMERIC FEATURES --------------------
def add_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = extract_numeric_features(df["text"])
    return pd.concat([df.reset_index(drop=True), num.reset_index(drop=True)], axis=1)


# -------------------- PIPELINE --------------------
def build_pipeline(max_features: int = 20000) -> Pipeline:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2
    )

    pre = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, "text_clean"),
            (
                "num",
                StandardScaler(with_mean=False),
                [
                    "sentiment",
                    "exclamation_count",
                    "all_caps_tokens",
                    "repeated_phrases",
                    "char_length",
                    "unique_word_ratio",
                ],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(max_iter=500, solver="liblinear")

    return Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])


# -------------------- TRAINING --------------------
def train(csv: Path, outdir: Path, test_size: float = 0.2, seed: int = 42) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    df = add_numeric(load_data(csv))

    X = df.drop(columns=["y", "text"])
    y = df["y"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)

    prob = pipe.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(yte, prob)),
        "avg_precision": float(average_precision_score(yte, prob)),
        "report": classification_report(
            yte,
            pred,
            target_names=["REAL", "FAKE"],
            output_dict=True,
        ),
    }

    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    joblib.dump(pipe, outdir / "pipeline.joblib")

    # -------------------- PLOTS --------------------
    import matplotlib
    matplotlib.use("Agg")

    # Confusion Matrix
    cm = confusion_matrix(yte, pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ROC Curve
    fig, ax = plt.subplots(figsize=(4, 3))
    RocCurveDisplay.from_predictions(yte, prob, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(outdir / "roc_curve.png", dpi=160)
    plt.close(fig)

    # PR Curve
    fig, ax = plt.subplots(figsize=(4, 3))
    PrecisionRecallDisplay.from_predictions(yte, prob, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(outdir / "pr_curve.png", dpi=160)
    plt.close(fig)

    return metrics


# -------------------- MAIN --------------------
def main():
    ap = argparse.ArgumentParser(description="Train Fake Review Detector")
    ap.add_argument("--csv", default="data/reviews_sample.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    metrics = train(
        Path(args.csv),
        Path(args.outdir),
        args.test_size,
        args.seed,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()