from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .model_builder import (
    apply_calibrator,
    fit_calibrator,
    heuristic_scores,
    predict_lexical,
    save_json,
    suspicious_threshold,
    threshold_metrics,
    tune_threshold,
)
from .utils import ensure_dir, load_config, resolve_path, set_seed


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(processed_dir / f"{name}.csv")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def select_embeddings(full_df: pd.DataFrame, subset_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    lookup = {text: i for i, text in enumerate(full_df["text"].tolist())}
    return embeddings[[lookup[text] for text in subset_df["text"].tolist()]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit calibration and thresholds after signal changes.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    set_seed(seed)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])

    log("Loading splits and artifacts...")
    val_df = load_split(processed_dir, "val")
    val_cal_df = load_split(processed_dir, "val_cal")
    val_tune_df = load_split(processed_dir, "val_tune")
    lexical = joblib.load(artifacts_dir / "lexical_model.joblib")
    semantic_head = None
    semantic_scaler = None
    head_path = artifacts_dir / "semantic_head.joblib"
    if head_path.exists():
        payload = joblib.load(head_path)
        semantic_head = payload["head"]
        semantic_scaler = payload["scaler"]

    val_embeddings_path = artifacts_dir / "embeddings" / "val.npy"
    val_embeddings = np.load(val_embeddings_path) if val_embeddings_path.exists() else None
    cal_embeddings = (
        select_embeddings(val_df, val_cal_df, val_embeddings)
        if val_embeddings is not None and semantic_head is not None else None
    )
    tune_embeddings = (
        select_embeddings(val_df, val_tune_df, val_embeddings)
        if val_embeddings is not None and semantic_head is not None else None
    )

    log(f"Scoring val_cal ({len(val_cal_df):,}) with updated signals...")
    lr_cal, svm_cal, _ = predict_lexical(lexical, val_cal_df["text"].tolist())
    sem_cal = (
        semantic_head.predict_proba(semantic_scaler.transform(cal_embeddings))[:, 1]
        if semantic_head is not None and cal_embeddings is not None else None
    )
    h_cal = heuristic_scores(val_cal_df["text"].tolist())

    log(f"Scoring val_tune ({len(val_tune_df):,}) with updated signals...")
    lr_tune, svm_tune, _ = predict_lexical(lexical, val_tune_df["text"].tolist())
    sem_tune = (
        semantic_head.predict_proba(semantic_scaler.transform(tune_embeddings))[:, 1]
        if semantic_head is not None and tune_embeddings is not None else None
    )
    h_tune = heuristic_scores(val_tune_df["text"].tolist())

    log("Fitting calibrator...")
    calibrator = fit_calibrator(
        lr_cal, svm_cal, sem_cal, h_cal,
        val_cal_df["label"].to_numpy(dtype=int),
        seed=seed,
        logger=log,
    )
    joblib.dump(calibrator, artifacts_dir / "calibrator.joblib", compress=3)

    final_tune = apply_calibrator(
        calibrator, lr_tune, svm_tune, sem_tune, h_tune
    )
    log("Tuning thresholds...")
    spam_result = tune_threshold(
        val_tune_df["label"].to_numpy(dtype=int),
        final_tune,
        fpr_target=float(config["model"]["calibration"]["fpr_target"]),
        logger=log,
    )
    suspicious_result = suspicious_threshold(
        val_tune_df["label"].to_numpy(dtype=int),
        final_tune,
        spam_result["threshold"],
        fpr_target=float(config["model"]["calibration"]["suspicious_fpr_target"]),
        logger=log,
    )
    thresholds = {
        "spam_threshold": round(float(spam_result["threshold"]), 6),
        "suspicious_threshold": round(float(suspicious_result["threshold"]), 6),
        "fpr_target": float(config["model"]["calibration"]["fpr_target"]),
        "suspicious_fpr_target": float(config["model"]["calibration"]["suspicious_fpr_target"]),
        "tuned_on": "val_tune",
        "val_tune_metrics": spam_result["metrics"],
        "suspicious_metrics": suspicious_result["metrics"],
        "risk_tiers": {
            "NORMAL": f"< {suspicious_result['threshold']:.4f}",
            "SUSPICIOUS": f"[{suspicious_result['threshold']:.4f}, {spam_result['threshold']:.4f})",
            "SPAM": f">= {spam_result['threshold']:.4f}",
        },
    }
    save_json(artifacts_dir / "threshold_config.json", thresholds)
    save_json(
        artifacts_dir / "val_tune_metrics.json",
        threshold_metrics(
            val_tune_df["label"].to_numpy(dtype=int),
            final_tune,
            thresholds["spam_threshold"],
        ),
    )

    metadata_path = artifacts_dir / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        metadata["thresholds"] = thresholds
        metadata["recalibrated_at"] = datetime.now(timezone.utc).isoformat()
        metadata["calibrator"] = {
            "feature_names": calibrator["feature_names"],
            "train_rows": calibrator["train_rows"],
            "train_prior": calibrator["train_prior"],
        }
        save_json(metadata_path, metadata)

    summary = {
        "recalibrated_at": datetime.now(timezone.utc).isoformat(),
        "calibrator_features": calibrator["feature_names"],
        "spam_threshold": thresholds["spam_threshold"],
        "suspicious_threshold": thresholds["suspicious_threshold"],
        "val_tune_metrics": spam_result["metrics"],
        "suspicious_metrics": suspicious_result["metrics"],
        "used_semantic": sem_cal is not None,
    }
    save_json(artifacts_dir / "recalibration_summary.json", summary)
    log(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
