from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.features import FEATURE_NAMES
from src.model_builder import (
    predict_lexical,
    suspicious_threshold,
    threshold_metrics,
    tune_threshold,
)
from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "static_space"
OUT.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    artifacts = ROOT / "artifacts"
    lexical = joblib.load(artifacts / "lexical_model.joblib")
    fe = lexical["feature_extractor"]
    lr = lexical["lr"]
    svm = lexical["svm"]

    val = pd.read_csv(ROOT / "data" / "processed" / "val_tune.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / "test_realistic.csv")

    val_lr, val_svm, _ = predict_lexical(lexical, val["text"].astype(str).tolist())
    test_lr, test_svm, _ = predict_lexical(lexical, test["text"].astype(str).tolist())
    y_val = val["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    spam_result = tune_threshold(y_val, val_lr, fpr_target=0.01, logger=print)
    suspicious_result = suspicious_threshold(
        y_val,
        val_lr,
        spam_result["threshold"],
        fpr_target=0.05,
        logger=print,
    )
    spam_threshold = float(spam_result["threshold"])
    suspicious_value = float(suspicious_result["threshold"])

    test_metrics = threshold_metrics(y_test, test_lr, spam_threshold)
    test_metrics.update({
        "pr_auc": round(float(average_precision_score(y_test, test_lr)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, test_lr)), 6),
        "brier": round(float(brier_score_loss(y_test, test_lr)), 6),
        "accuracy": round(float(accuracy_score(y_test, (test_lr >= spam_threshold).astype(int))), 6),
    })

    n_char = int(fe.n_char_)
    n_word = int(fe.n_word_)
    n_signal = int(fe.n_signal_)
    assert n_char + n_word + n_signal == len(lr.coef_[0])

    model_payload = {
        "version": "1.0.0-browser-lexical",
        "n_char": n_char,
        "n_word": n_word,
        "n_signal": n_signal,
        "feature_count": int(fe.n_features_),
        "char_ngram_range": list(fe.char_ngram_range),
        "word_ngram_range": list(fe.word_ngram_range),
        "sublinear_tf": bool(fe.sublinear_tf),
        "char_ngrams": fe.char_.get_feature_names_out().tolist(),
        "word_ngrams": fe.word_.get_feature_names_out().tolist(),
        "char_idf": fe.char_.idf_.astype(np.float32).tolist(),
        "word_idf": fe.word_.idf_.astype(np.float32).tolist(),
        "char_lr": lr.coef_[0][:n_char].astype(np.float32).tolist(),
        "word_lr": lr.coef_[0][n_char:n_char + n_word].astype(np.float32).tolist(),
        "signal_lr": lr.coef_[0][n_char + n_word:].astype(np.float32).tolist(),
        "char_svm": svm.coef_[0][:n_char].astype(np.float32).tolist(),
        "word_svm": svm.coef_[0][n_char:n_char + n_word].astype(np.float32).tolist(),
        "signal_svm": svm.coef_[0][n_char + n_word:].astype(np.float32).tolist(),
        "lr_intercept": float(lr.intercept_[0]),
        "svm_intercept": float(svm.intercept_[0]),
        "signal_names": FEATURE_NAMES,
        "heuristic_index": FEATURE_NAMES.index("heuristic_score"),
    }
    raw = json.dumps(model_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    (OUT / "browser_model.json.gz").write_bytes(gzip.compress(raw, compresslevel=9))
    print(f"model json {len(raw)/1e6:.1f} MB, gzip {(OUT / 'browser_model.json.gz').stat().st_size/1e6:.1f} MB")

    thresholds = {
        "spam_threshold": round(spam_threshold, 6),
        "suspicious_threshold": round(suspicious_value, 6),
        "fpr_target": 0.01,
        "suspicious_fpr_target": 0.05,
        "tuned_on": "val_tune",
        "val_tune_metrics": spam_result["metrics"],
        "suspicious_metrics": suspicious_result["metrics"],
        "risk_tiers": {
            "NORMAL": f"< {suspicious_value:.4f}",
            "SUSPICIOUS": f"[{suspicious_value:.4f}, {spam_threshold:.4f})",
            "SPAM": f">= {spam_threshold:.4f}",
        },
    }
    write_json(OUT / "browser_thresholds.json", thresholds)

    metadata = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))
    metadata["model_type"] = "browser lexical + signals + rules"
    metadata["browser_mode"] = {
        "branch": "lexical and signal layer",
        "semantic_encoder": "not loaded in the free browser demo",
        "reason": "Hugging Face free Docker Spaces require PRO; the free Space runs the Python model in the browser.",
    }
    metadata["lexical"] = {
        "best_C": lexical.get("best_C"),
        "svm_C": lexical.get("svm_C"),
        "feature_count": lexical.get("feature_count"),
        "train_rows": lexical.get("train_rows"),
        "val_metrics": lexical.get("val_metrics"),
    }

    evaluation = {
        "test_realistic": {
            "prior": round(float(y_test.mean()), 6),
            "threshold": round(spam_threshold, 6),
            "split": "test_realistic",
            "browser_mode": "lexical only",
            "metrics": test_metrics,
        }
    }
    dataset = json.loads((ROOT / "data" / "processed" / "dataset_stats.json").read_text(encoding="utf-8"))
    stats = {
        "model": metadata,
        "thresholds": thresholds,
        "semantic": {
            "model_name": "not loaded in the free browser demo",
            "backend": "pyodide-lexical",
            "embedding_dim": 0,
        },
        "evaluation": evaluation,
        "dataset": dataset,
        "onnx_benchmark": None,
    }
    write_json(OUT / "browser_stats.json", stats)

    health = {
        "status": "ok",
        "model_version": "1.0.0-browser-lexical",
        "model_type": "browser lexical + signals + rules",
        "backbone": None,
        "semantic_backend": "pyodide-lexical",
        "semantic_error": None,
        "thresholds": {
            "spam_threshold": round(spam_threshold, 6),
            "suspicious_threshold": round(suspicious_value, 6),
        },
        "device": "browser",
        "artifacts_dir": "browser",
    }
    write_json(OUT / "browser_health.json", health)
    examples = {"examples": list(config.get("ui", {}).get("examples", []))}
    write_json(OUT / "browser_examples.json", examples)

    print(json.dumps({
        "spam_threshold": spam_threshold,
        "suspicious_threshold": suspicious_value,
        "test_pr_auc": test_metrics["pr_auc"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_fpr": test_metrics["fpr"],
        "test_brier": test_metrics["brier"],
    }, indent=2))


if __name__ == "__main__":
    main()
