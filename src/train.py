from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .model_builder import (
    SemanticEncoder,
    apply_calibrator,
    fit_calibrator,
    heuristic_scores,
    predict_lexical,
    save_json,
    stratified_indices,
    suspicious_threshold,
    threshold_metrics,
    train_lexical_models,
    train_semantic_head,
    tune_threshold,
)
from .utils import ensure_dir, load_config, resolve_path, set_seed


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    path = processed_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}. Run src.data_pipeline first.")
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def select_embeddings(
    full_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    full_embeddings: np.ndarray,
) -> np.ndarray:
    if len(subset_df) == 0:
        return np.empty((0, full_embeddings.shape[1]), dtype=np.float32)
    lookup = {text: i for i, text in enumerate(full_df["text"].tolist())}
    rows = [lookup[text] for text in subset_df["text"].tolist()]
    return full_embeddings[rows]


def semantic_probs_from_embeddings(semantic: dict, embeddings: np.ndarray) -> np.ndarray:
    return semantic["head"].predict_proba(
        semantic["scaler"].transform(embeddings)
    )[:, 1]


def build_final_probs(
    lexical: dict,
    semantic: dict | None,
    calibrator: dict,
    df: pd.DataFrame,
    embeddings: np.ndarray | None,
) -> dict:
    texts = df["text"].tolist()
    lr_probs, svm_scores, _ = predict_lexical(lexical, texts)
    sem_probs = None
    if semantic is not None:
        if embeddings is None:
            embeddings = semantic["encoder"].encode(texts)
        sem_probs = semantic_probs_from_embeddings(semantic, embeddings)
    h = heuristic_scores(texts)
    final = apply_calibrator(calibrator, lr_probs, svm_scores, sem_probs, h)
    return {
        "final": final,
        "lexical_lr": lr_probs,
        "lexical_svm": svm_scores,
        "semantic": sem_probs,
        "heuristic": h,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multilingual spam detector.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--reuse-lexical", action="store_true")
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument("--semantic-max-length", type=int, default=None)
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-semantic-train", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    set_seed(seed)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])
    embeddings_dir = ensure_dir(artifacts_dir / "embeddings")

    log("Loading processed splits...")
    train_df = load_split(processed_dir, "train")
    val_df = load_split(processed_dir, "val")
    val_cal_df = load_split(processed_dir, "val_cal")
    val_tune_df = load_split(processed_dir, "val_tune")
    test_df = load_split(processed_dir, "test")
    test_real_df = load_split(processed_dir, "test_realistic")
    if args.max_train and args.max_train < len(train_df):
        idx = stratified_indices(
            train_df["label"].to_numpy(),
            args.max_train,
            seed,
            strata=train_df["language"].to_numpy(),
        )
        train_df = train_df.iloc[idx].reset_index(drop=True)
    log(
        f"Rows: train={len(train_df):,}, val={len(val_df):,}, "
        f"val_cal={len(val_cal_df):,}, val_tune={len(val_tune_df):,}, "
        f"test={len(test_df):,}, test_real={len(test_real_df):,}"
    )

    if args.semantic_only:
        args.reuse_lexical = True
        args.skip_semantic = False
    if args.reuse_lexical:
        lexical_path = artifacts_dir / "lexical_model.joblib"
        if not lexical_path.exists():
            raise FileNotFoundError(
                f"Cannot reuse lexical model: {lexical_path} does not exist."
            )
        log(f"Reusing lexical model from {lexical_path}")
        lexical = joblib.load(lexical_path)
    else:
        log("Training hybrid lexical model...")
        lexical = train_lexical_models(train_df, val_df, config, logger=log)
        joblib.dump(lexical, artifacts_dir / "lexical_model.joblib", compress=3)
        log(f"Saved lexical model ({lexical['feature_count']:,} features)")

    semantic = None
    val_embeddings = None
    test_embeddings = None
    train_sample_df = None
    if not args.skip_semantic:
        scfg = config["model"]["semantic"]
        max_sem = int(
            args.max_semantic_train or scfg.get("max_train_samples", 56000)
        )
        sem_idx = stratified_indices(
            train_df["label"].to_numpy(),
            min(max_sem, len(train_df)),
            seed,
            strata=train_df["language"].to_numpy(),
        )
        train_sample_df = train_df.iloc[sem_idx].reset_index(drop=True)
        log(f"Encoding {len(train_sample_df):,} language-stratified semantic training rows...")
        onnx_int8 = artifacts_dir / "semantic_encoder_int8.onnx"
        onnx_fp32 = artifacts_dir / "semantic_encoder.onnx"
        onnx_path = onnx_int8 if onnx_int8.exists() else (
            onnx_fp32 if onnx_fp32.exists() else None
        )
        tokenizer_dir = artifacts_dir / "tokenizer"
        semantic_max_length = int(
            args.semantic_max_length or scfg.get("max_seq_length", 128)
        )
        encoder = SemanticEncoder(
            model_name=scfg["backbone"],
            onnx_path=onnx_path,
            tokenizer_dir=tokenizer_dir if tokenizer_dir.exists() else None,
            max_seq_length=semantic_max_length,
            batch_size=int(scfg.get("batch_size", 64)),
        )
        log(
            f"Semantic backend: {'ONNX ' + onnx_path.name if onnx_path else 'PyTorch'} "
            f"(max_seq_length={semantic_max_length})"
        )
        train_embeddings = encoder.encode_to_file(
            train_sample_df["text"].tolist(),
            embeddings_dir / "train.npy",
            show_progress=True,
        )
        log(f"Train embeddings: {train_embeddings.shape}")
        log(f"Encoding {len(val_df):,} validation rows...")
        val_embeddings = encoder.encode_to_file(
            val_df["text"].tolist(),
            embeddings_dir / "val.npy",
            show_progress=True,
        )
        log(f"Validation embeddings: {val_embeddings.shape}")
        log(f"Encoding {len(test_df):,} test rows...")
        test_embeddings = encoder.encode_to_file(
            test_df["text"].tolist(),
            embeddings_dir / "test.npy",
            show_progress=True,
        )
        log(f"Test embeddings: {test_embeddings.shape}")

        log("Training calibrated semantic head...")
        semantic = train_semantic_head(
            train_embeddings,
            train_sample_df["label"].to_numpy(dtype=int),
            val_embeddings,
            val_df["label"].to_numpy(dtype=int),
            config,
            logger=log,
        )
        semantic.update({
            "encoder": encoder,
            "model_name": scfg["backbone"],
            "max_seq_length": semantic_max_length,
            "batch_size": int(scfg.get("batch_size", 64)),
            "train_sample_rows": int(len(train_sample_df)),
            "val_rows": int(len(val_df)),
        })
        head_payload = {
            "head": semantic["head"],
            "scaler": semantic["scaler"],
            "best_C": semantic["best_C"],
            "grid_results": semantic["grid_results"],
            "val_metrics": semantic["val_metrics"],
            "embedding_dim": semantic["embedding_dim"],
            "model_name": semantic["model_name"],
            "max_seq_length": semantic["max_seq_length"],
            "batch_size": semantic["batch_size"],
        }
        joblib.dump(head_payload, artifacts_dir / "semantic_head.joblib", compress=3)
        save_json(artifacts_dir / "semantic_metadata.json", {
            "model_name": semantic["model_name"],
            "max_seq_length": semantic["max_seq_length"],
            "batch_size": semantic["batch_size"],
            "embedding_dim": semantic["embedding_dim"],
            "best_C": semantic["best_C"],
            "grid_results": semantic["grid_results"],
            "val_metrics": semantic["val_metrics"],
        })
        log("Saved semantic head and metadata")

    else:
        log("Skipping semantic branch by request")

    log("Selecting calibration/tuning embeddings...")
    val_cal_embeddings = (
        select_embeddings(val_df, val_cal_df, val_embeddings)
        if semantic is not None else None
    )
    val_tune_embeddings = (
        select_embeddings(val_df, val_tune_df, val_embeddings)
        if semantic is not None else None
    )
    test_real_embeddings = (
        select_embeddings(test_df, test_real_df, test_embeddings)
        if semantic is not None else None
    )

    log("Fitting final probability calibrator on val_cal...")
    lr_cal, svm_cal, _ = predict_lexical(lexical, val_cal_df["text"].tolist())
    sem_cal = (
        semantic_probs_from_embeddings(semantic, val_cal_embeddings)
        if semantic is not None else None
    )
    h_cal = heuristic_scores(val_cal_df["text"].tolist())
    calibrator = fit_calibrator(
        lr_cal,
        svm_cal,
        sem_cal,
        h_cal,
        val_cal_df["label"].to_numpy(dtype=int),
        seed=seed,
        logger=log,
    )
    joblib.dump(calibrator, artifacts_dir / "calibrator.joblib", compress=3)

    log("Tuning thresholds on the disjoint val_tune split...")
    tune = build_final_probs(
        lexical, semantic, calibrator, val_tune_df, val_tune_embeddings
    )
    spam_t = tune_threshold(
        val_tune_df["label"].to_numpy(dtype=int),
        tune["final"],
        fpr_target=float(config["model"]["calibration"]["fpr_target"]),
        logger=log,
    )
    suspicious = suspicious_threshold(
        val_tune_df["label"].to_numpy(dtype=int),
        tune["final"],
        spam_t["threshold"],
        fpr_target=float(config["model"]["calibration"]["suspicious_fpr_target"]),
        logger=log,
    )
    thresholds = {
        "spam_threshold": round(float(spam_t["threshold"]), 6),
        "suspicious_threshold": round(float(suspicious["threshold"]), 6),
        "fpr_target": float(config["model"]["calibration"]["fpr_target"]),
        "suspicious_fpr_target": float(config["model"]["calibration"]["suspicious_fpr_target"]),
        "tuned_on": "val_tune",
        "val_tune_metrics": spam_t["metrics"],
        "suspicious_metrics": suspicious["metrics"],
        "risk_tiers": {
            "NORMAL": f"< {suspicious['threshold']:.4f}",
            "SUSPICIOUS": f"[{suspicious['threshold']:.4f}, {spam_t['threshold']:.4f})",
            "SPAM": f">= {spam_t['threshold']:.4f}",
        },
    }
    save_json(artifacts_dir / "threshold_config.json", thresholds)
    log(f"Thresholds: {json.dumps(thresholds['risk_tiers'], ensure_ascii=False)}")

    val_metrics_at_threshold = threshold_metrics(
        val_tune_df["label"].to_numpy(dtype=int),
        tune["final"],
        thresholds["spam_threshold"],
    )
    save_json(artifacts_dir / "val_tune_metrics.json", val_metrics_at_threshold)

    metadata = {
        "project": config["project"]["name"],
        "version": config["project"]["version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model_type": "hybrid_tfidf_signals_plus_multilingual_minilm",
        "backbone": semantic["model_name"] if semantic is not None else None,
        "lexical": {
            "best_C": lexical["best_C"],
            "svm_C": lexical["svm_C"],
            "feature_count": lexical["feature_count"],
            "train_rows": lexical["train_rows"],
            "grid_results": lexical["grid_results"],
            "val_metrics": lexical["val_metrics"],
        },
        "semantic": {
            "model_name": semantic["model_name"],
            "embedding_dim": semantic["embedding_dim"],
            "train_sample_rows": semantic["train_sample_rows"],
            "val_rows": semantic["val_rows"],
            "best_C": semantic["best_C"],
            "grid_results": semantic["grid_results"],
            "val_metrics": semantic["val_metrics"],
        } if semantic is not None else None,
        "calibrator": {
            "feature_names": calibrator["feature_names"],
            "train_rows": calibrator["train_rows"],
            "train_prior": calibrator["train_prior"],
        },
        "thresholds": thresholds,
        "val_tune_metrics": val_metrics_at_threshold,
        "data": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "val_cal_rows": int(len(val_cal_df)),
            "val_tune_rows": int(len(val_tune_df)),
            "test_rows": int(len(test_df)),
            "test_realistic_rows": int(len(test_real_df)),
        },
        "notes": [
            "The production CPU model uses a frozen multilingual MiniLM encoder plus a calibrated linear head, not the legacy English-only DeBERTa-v3-extra-small H5 model.",
            "Thresholds were tuned on a 20% prior validation split disjoint from the calibration split.",
            "The full two-stage Keras multilingual DeBERTa fine-tuning path is available in src/keras_transformer.py for GPU environments.",
        ],
    }
    save_json(artifacts_dir / "model_metadata.json", metadata)

    log("Quick sanity predictions:")
    sanity = [
        "Give me $5 bucks I will give you an RTX 3090",
        "Bro can you send me $5 for lunch? I will pay you back tomorrow.",
        "Your OTP is 482913. Do not share it with anyone.",
        "URGENT: Your bKash account is suspended. Click http://bkash-verify.xyz to reactivate.",
        "আপনি ৫,০০,০০০ টাকা জিতেছেন! পুরস্কার পেতে আপনার বিকাশ নম্বর পাঠান।",
        "Aapka KYC pending hai, turant is link par click karke update karein: bit[.]ly/kyc-update",
        "Mi cuenta ha sido bloqueada, verifique aquí: http://bbva-seguro.xyz",
        "Votre compte est suspendu, cliquez ici: http://paypal-verification.top",
        "حسابك موقوف، انقر هنا: http://bank-verify.xyz",
    ]
    sanity_emb = (
        semantic["encoder"].encode(sanity)
        if semantic is not None else None
    )
    sanity_probs = build_final_probs(
        lexical, semantic, calibrator, pd.DataFrame({"text": sanity}), sanity_emb
    )["final"]
    for text, p in zip(sanity, sanity_probs):
        tier = "SPAM" if p >= thresholds["spam_threshold"] else (
            "SUSPICIOUS" if p >= thresholds["suspicious_threshold"] else "NORMAL"
        )
        log(f"  {tier:10s} p={p:.4f} | {text[:100]}")

    log("Training complete.")
    log(f"Artifacts: {artifacts_dir}")
    del lexical
    gc.collect()


if __name__ == "__main__":
    main()
