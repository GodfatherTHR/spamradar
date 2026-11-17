from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .inference import SpamPredictor
from .model_builder import (
    suspicious_threshold,
    threshold_metrics,
    tune_threshold,
)
from .utils import ensure_dir, load_config, resolve_path, set_seed, text_hash

TARGET_LANGUAGES = {
    "English", "Bengali", "Banglish", "Hindi", "Hinglish",
    "Spanish", "French", "Arabic", "mixed",
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    predicted = (probs >= 0.5).astype(int)
    confidence = np.where(predicted == 1, probs, 1.0 - probs)
    correctness = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    if total == 0:
        return 0.0
    score = 0.0
    for i in range(bins):
        lower, upper = edges[i], edges[i + 1]
        mask = (confidence >= lower) & (confidence < upper if i < bins - 1 else confidence <= upper)
        if not mask.any():
            continue
        score += (mask.sum() / total) * abs(correctness[mask].mean() - confidence[mask].mean())
    return float(score)


def metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    base = threshold_metrics(y_true, probs, threshold)
    pred = (probs >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) > 1
    result = {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y_true, pred)), 6),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 6),
        "fpr": round(float(base["fpr"]), 6),
        "fnr": round(float(base["fnr"]), 6),
        "tp": int(base["tp"]),
        "fp": int(base["fp"]),
        "tn": int(base["tn"]),
        "fn": int(base["fn"]),
        "predicted_spam": int(pred.sum()),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 6) if both_classes else None,
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 6) if both_classes else None,
        "brier": round(float(brier_score_loss(y_true, probs)), 6),
        "log_loss": round(float(log_loss(y_true, np.clip(probs, 1e-9, 1 - 1e-9), labels=[0, 1])), 6),
        "ece": round(ece_score(y_true, probs), 6),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "rows": int(len(y_true)),
        "spam_prior": round(float(np.mean(y_true)), 6),
    }
    return result


def category_group(category: str, label: int) -> str:
    value = str(category or "").casefold()
    if int(label) == 0:
        if any(k in value for k in ("otp", "bank", "service", "kyc", "transaction", "reminder")):
            return "Hard negative · OTP / bank / service"
        if "delivery" in value or "courier" in value:
            return "Hard negative · delivery"
        if "marketplace" in value:
            return "Hard negative · marketplace"
        if "promo" in value or "promotional" in value:
            return "Hard negative · opt-in promo"
        if "personal" in value or "benign" in value:
            return "Hard negative · personal / benign"
        return "Hard negative · other"
    if any(k in value for k in ("p2p", "crypto", "marketplace", "romance")):
        return "Scam · P2P / crypto / marketplace / romance"
    if any(k in value for k in ("bank", "kyc", "upi", "loan", "investment", "freeze")):
        return "Scam · bank / KYC / UPI / investment"
    if any(k in value for k in ("delivery", "courier", "otp", "parcel")):
        return "Scam · delivery / OTP"
    if any(k in value for k in ("lottery", "prize", "promo")):
        return "Scam / spam · lottery / promotional"
    if any(k in value for k in ("job", "telegram", "hiring")):
        return "Scam · job"
    if "phishing" in value:
        return "Scam · phishing"
    return "Scam · other"


def language_group(language: str) -> str:
    value = str(language or "other")
    return value if value in TARGET_LANGUAGES else "Other / rare"


def per_group(df: pd.DataFrame, probs: np.ndarray, threshold: float, group_col: str) -> list[dict]:
    work = df.copy()
    work["_prob"] = np.asarray(probs, dtype=float)
    work["_pred"] = (work["_prob"] >= threshold).astype(int)
    rows = []
    for group, subset in work.groupby(group_col, sort=False):
        y = subset["label"].to_numpy(dtype=int)
        prob = subset["_prob"].to_numpy(dtype=float)
        pred = subset["_pred"].to_numpy(dtype=int)
        both = len(np.unique(y)) > 1
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({
            "group": str(group),
            "rows": int(len(subset)),
            "spam": int(y.sum()),
            "ham": int((y == 0).sum()),
            "spam_prior": round(float(y.mean()), 4),
            "predicted_spam": int(pred.sum()),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "fpr": round(float(fpr), 4),
            "pr_auc": round(float(average_precision_score(y, prob)), 4) if both else None,
            "roc_auc": round(float(roc_auc_score(y, prob)), 4) if both else None,
            "brier": round(float(brier_score_loss(y, prob)), 4),
            "ece": round(float(ece_score(y, prob)), 4),
        })
    return sorted(rows, key=lambda r: r["rows"], reverse=True)


def select_embeddings(full_df: pd.DataFrame, subset_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray | None:
    if embeddings is None:
        return None
    lookup = {text: i for i, text in enumerate(full_df["text"].tolist())}
    try:
        return embeddings[[lookup[text] for text in subset_df["text"].tolist()]]
    except KeyError:
        return None


def load_embeddings(artifacts_dir: Path, name: str) -> np.ndarray | None:
    path = artifacts_dir / "embeddings" / f"{name}.npy"
    if path.exists():
        try:
            return np.load(path)
        except Exception:
            return None
    return None


def load_adversarial(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    df = pd.DataFrame(rows)
    if "text" not in df:
        raise ValueError("adversarial_suite.json must contain a 'text' field")
    if "label" in df:
        label_series = df["label"]
    elif "expected_label" in df:
        label_series = df["expected_label"]
    else:
        raise ValueError("adversarial_suite.json must contain 'label' or 'expected_label'")
    mapping = {
        "spam": 1, "scam": 1, "phishing": 1, "1": 1, 1: 1, True: 1,
        "ham": 0, "normal": 0, "legit": 0, "legitimate": 0, "0": 0, 0: 0, False: 0,
    }
    df["label"] = label_series.map(lambda x: mapping.get(x, mapping.get(str(x).casefold(), 0))).astype(int)
    df["language"] = df.get("language", "unknown").fillna("unknown")
    df["category"] = df.get("category", "unknown").fillna("unknown")
    df["why"] = df.get("why", "").fillna("")
    return df


def adversarial_report(
    predictor: SpamPredictor,
    suite: pd.DataFrame,
    spam_threshold: float,
    suspicious_threshold_value: float,
) -> dict:
    rows = []
    hard_pass = []
    block_safe = []
    soft_pass = []
    for i, row in suite.reset_index(drop=True).iterrows():
        prediction = predictor.predict(row["text"])
        expected = int(row["label"])
        probability = float(prediction["spam_probability"])
        tier = prediction["label"]
        predicted = int(tier == "SPAM")
        is_suspicious = tier == "SUSPICIOUS"
        hard_ok = (expected == 1 and tier == "SPAM") or (expected == 0 and tier == "NORMAL")
        safe_ok = not (expected == 0 and predicted == 1)
        soft_ok = (expected == 1 and tier in {"SPAM", "SUSPICIOUS"}) or (expected == 0 and tier == "NORMAL")
        hard_pass.append(hard_ok)
        block_safe.append(safe_ok)
        soft_pass.append(soft_ok)
        decision = prediction.get("decision", {})
        rows.append({
            "id": row.get("id", f"case_{i}"),
            "text": row["text"],
            "language": row["language"],
            "category": row["category"],
            "expected": "SPAM" if expected == 1 else "NORMAL",
            "predicted": tier,
            "spam_probability": round(probability, 6),
            "hard_pass": bool(hard_ok),
            "block_safe": bool(safe_ok),
            "soft_pass": bool(soft_ok),
            "why": row["why"],
            "decision_source": decision.get("source", "calibrated_model"),
            "rule_reasons": decision.get("rule_reasons", []),
            "signals": [s.get("title") for s in prediction.get("detected_signals", [])[:4]],
        })
    failures = [r for r in rows if not r["hard_pass"]]
    by_language = {}
    for row in rows:
        key = language_group(row["language"])
        bucket = by_language.setdefault(key, {"total": 0, "hard_pass": 0, "block_safe": 0})
        bucket["total"] += 1
        bucket["hard_pass"] += int(row["hard_pass"])
        bucket["block_safe"] += int(row["block_safe"])
    for bucket in by_language.values():
        bucket["hard_pass_rate"] = round(bucket["hard_pass"] / max(1, bucket["total"]), 4)
        bucket["block_safe_rate"] = round(bucket["block_safe"] / max(1, bucket["total"]), 4)
    return {
        "total": len(rows),
        "hard_pass": int(sum(hard_pass)),
        "hard_pass_rate": round(float(np.mean(hard_pass)) if hard_pass else 0.0, 4),
        "soft_pass": int(sum(soft_pass)),
        "soft_pass_rate": round(float(np.mean(soft_pass)) if soft_pass else 0.0, 4),
        "block_safe": int(sum(block_safe)),
        "block_safe_rate": round(float(np.mean(block_safe)) if block_safe else 0.0, 4),
        "false_blocks": sum(1 for r in rows if r["expected"] == "NORMAL" and r["predicted"] == "SPAM"),
        "false_reviews": sum(1 for r in rows if r["expected"] == "NORMAL" and r["predicted"] == "SUSPICIOUS"),
        "missed_scams": sum(1 for r in rows if r["expected"] == "SPAM" and r["predicted"] == "NORMAL"),
        "decision_layer": "calibrated model + high-confidence rule escalation",
        "raw_thresholds": {
            "spam": spam_threshold,
            "suspicious": suspicious_threshold_value,
        },
        "by_language": by_language,
        "failures": failures,
        "all_cases": rows,
    }


def save_plots(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    per_language: list[dict],
    plots_dir: Path,
) -> dict:
    ensure_dir(plots_dir)
    paths = {}
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#0b1424", "figure.facecolor": "#070c15",
        "grid.color": "#1c2a42", "text.color": "#dbe5f5",
        "axes.labelcolor": "#b8c6dc", "xtick.color": "#8fa0ba", "ytick.color": "#8fa0ba",
    })

    cm = confusion_matrix(y_true, (probs >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=160)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="mako", cbar=False,
        xticklabels=["NORMAL", "SPAM"], yticklabels=["NORMAL", "SPAM"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix · threshold={threshold:.4f}")
    path = plots_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths["confusion_matrix"] = str(path)

    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=160)
    ax.plot(recall, precision, color="#45e3c1", linewidth=2, label=f"PR-AUC={pr_auc:.4f}")
    ax.fill_between(recall, precision, alpha=0.12, color="#45e3c1")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve")
    ax.legend(loc="lower left")
    path = plots_dir / "pr_curve.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths["pr_curve"] = str(path)

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=160)
    ax.plot(fpr, tpr, color="#6ea8ff", linewidth=2, label=f"ROC-AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#3b4c68", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    path = plots_dir / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths["roc_curve"] = str(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    ax.hist(probs[y_true == 0], bins=40, alpha=0.65, color="#41dfa0", label="NORMAL", density=True)
    ax.hist(probs[y_true == 1], bins=40, alpha=0.65, color="#ff5f70", label="SPAM", density=True)
    ax.axvline(threshold, color="#f7b955", linestyle="--", linewidth=1.5, label="Spam threshold")
    ax.set_xlabel("Calibrated spam probability")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution")
    ax.legend(loc="upper center")
    path = plots_dir / "score_distribution.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths["score_distribution"] = str(path)

    fraction_positive, mean_predicted = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=160)
    ax.plot(mean_predicted, fraction_positive, marker="o", color="#f7b955", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#3b4c68", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed spam rate")
    ax.set_title("Calibration curve")
    path = plots_dir / "calibration_curve.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths["calibration_curve"] = str(path)

    languages = [row for row in per_language if row["group"] not in {"Other / rare"}][:10]
    if languages:
        fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=160)
        names = [row["group"] for row in languages]
        values = [row["f1"] for row in languages]
        ax.barh(list(reversed(names)), list(reversed(values)), color="#45e3c1", alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1 score at tuned threshold")
        ax.set_title("Per-language F1")
        path = plots_dir / "per_language_f1.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths["per_language_f1"] = str(path)
    return paths


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    evaluation: dict,
    metadata: dict,
    adversarial: dict,
    thresholds: dict,
) -> None:
    lines = [
        "# Evaluation report — Multilingual Spam & Scam Detection",
        "",
        f"Generated: `{evaluation['generated_at']}`",
        "",
        "## Model",
        "",
        f"- Type: **{metadata.get('model_type', 'hybrid')}**",
        f"- Multilingual backbone: **{(metadata.get('semantic') or {}).get('model_name', 'lexical only')}**",
        f"- Spam threshold: **{thresholds.get('spam_threshold')}** (FPR target ≤ {thresholds.get('fpr_target')})",
        f"- Review threshold: **{thresholds.get('suspicious_threshold')}**",
        "",
        "## Held-out test metrics",
        "",
    ]
    for name, label in (("test_realistic", "Realistic 20% spam prior"), ("test", "Full held-out test")):
        block = evaluation.get(name, {})
        metrics = block.get("metrics", {})
        lines.append(f"### {label}")
        lines.append("")
        lines.append(markdown_table([metrics], [
            ("rows", "Rows"), ("spam_prior", "Spam prior"), ("pr_auc", "PR-AUC"),
            ("roc_auc", "ROC-AUC"), ("f1", "F1"), ("precision", "Precision"),
            ("recall", "Recall"), ("fpr", "FPR"), ("brier", "Brier"),
            ("ece", "ECE"), ("accuracy", "Accuracy"),
        ]))
        lines.append("")
        lines.append(f"Confusion matrix (rows=actual, cols=predicted): `{metrics.get('confusion_matrix')}`")
        lines.append("")

    lines.extend(["## Per-language breakdown", ""])
    lang_rows = evaluation.get("test_realistic", {}).get("per_language", [])
    lines.append(markdown_table(lang_rows, [
        ("group", "Language"), ("rows", "Rows"), ("spam_prior", "Spam prior"),
        ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
        ("fpr", "FPR"), ("pr_auc", "PR-AUC"),
    ]))
    lines.extend(["", "## Per-category breakdown", ""])
    cat_rows = evaluation.get("test_realistic", {}).get("per_category", [])
    lines.append(markdown_table(cat_rows, [
        ("group", "Category"), ("rows", "Rows"), ("spam_prior", "Spam prior"),
        ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
        ("fpr", "FPR"), ("pr_auc", "PR-AUC"),
    ]))
    lines.extend([
        "",
        "## Adversarial suite",
        "",
        f"- Cases: **{adversarial['total']}**",
        f"- Hard pass rate: **{adversarial['hard_pass_rate']:.2%}**",
        f"- Soft pass rate (review counts for scam): **{adversarial['soft_pass_rate']:.2%}**",
        f"- Block-safe rate (no normal message auto-blocked): **{adversarial['block_safe_rate']:.2%}**",
        f"- False blocks: **{adversarial['false_blocks']}**",
        f"- False reviews of normal messages: **{adversarial['false_reviews']}**",
        f"- Missed scams (predicted NORMAL): **{adversarial['missed_scams']}**",
        "",
    ])
    if adversarial["failures"]:
        lines.append("### Adversarial failures")
        lines.append("")
        for failure in adversarial["failures"][:25]:
            lines.append(
                f"- **{failure['id']}** [{failure['expected']} → {failure['predicted']}, "
                f"p={failure['spam_probability']:.3f}, {failure['language']}] "
                f"{failure['text'][:220]}"
            )
        lines.append("")
    lines.extend([
        "## Plots",
        "",
        "- `artifacts/plots/confusion_matrix.png`",
        "- `artifacts/plots/pr_curve.png`",
        "- `artifacts/plots/roc_curve.png`",
        "- `artifacts/plots/score_distribution.png`",
        "- `artifacts/plots/calibration_curve.png`",
        "- `artifacts/plots/per_language_f1.png`",
        "",
        "## Notes and limitations",
        "",
        "- The headline operating point is precision-constrained: the spam threshold is tuned for FPR ≤ 1% on a disjoint validation split.",
        "- The full test set has a ~45% spam prior because it is built from merged spam/ham corpora. The realistic subset resamples it to 20% for a more deployment-like view.",
        "- SMS datasets contain label noise. Obvious contradictory noisy-source rows were filtered and ambiguous dedupe clusters were dropped, but some noise remains.",
        "- Text-only detection cannot resolve every ambiguous case (for example, a genuine \"I sent money by mistake, please return it\" versus the same message sent by a scammer). Use the SUSPICIOUS tier for review.",
        "- The legacy English-only DeBERTa-v3-extra-small H5 model is not used for multilingual inference.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the multilingual spam detector.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])
    plots_dir = ensure_dir(artifacts_dir / "plots")

    log("Loading predictor...")
    predictor = SpamPredictor()
    thresholds = predictor.thresholds
    spam_t = float(thresholds["spam_threshold"])
    suspicious_t = float(thresholds["suspicious_threshold"])
    log(f"Thresholds: spam={spam_t:.4f}, suspicious={suspicious_t:.4f}")

    test_df = pd.read_csv(processed_dir / "test.csv")
    test_real_df = pd.read_csv(processed_dir / "test_realistic.csv")
    for frame in (test_df, test_real_df):
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["label"] = frame["label"].astype(int)
        frame["category_group"] = [
            category_group(c, l) for c, l in zip(frame["category"], frame["label"])
        ]
        frame["language_group"] = frame["language"].map(language_group)

    test_embeddings = load_embeddings(artifacts_dir, "test")
    test_real_embeddings = (
        select_embeddings(test_df, test_real_df, test_embeddings)
        if test_embeddings is not None else None
    )

    log(f"Scoring {len(test_df):,} test rows...")
    test_scores = predictor.score_texts(test_df["text"].tolist(), embeddings=test_embeddings)
    test_probs = test_scores["final"]

    log(f"Scoring {len(test_real_df):,} realistic-prior test rows...")
    real_scores = predictor.score_texts(test_real_df["text"].tolist(), embeddings=test_real_embeddings)
    real_probs = real_scores["final"]

    if args.retune:
        log("Re-tuning thresholds on validation split...")
        val_df = pd.read_csv(processed_dir / "val.csv")
        val_tune_df = pd.read_csv(processed_dir / "val_tune.csv")
        for frame in (val_df, val_tune_df):
            frame["text"] = frame["text"].fillna("").astype(str)
            frame["label"] = frame["label"].astype(int)
        val_embeddings = load_embeddings(artifacts_dir, "val")
        val_tune_embeddings = (
            select_embeddings(val_df, val_tune_df, val_embeddings)
            if val_embeddings is not None else None
        )
        tune_scores = predictor.score_texts(
            val_tune_df["text"].tolist(), embeddings=val_tune_embeddings
        )
        spam_result = tune_threshold(
            val_tune_df["label"].to_numpy(dtype=int),
            tune_scores["final"],
            fpr_target=float(thresholds.get("fpr_target", 0.01)),
            logger=log,
        )
        suspicious_result = suspicious_threshold(
            val_tune_df["label"].to_numpy(dtype=int),
            tune_scores["final"],
            spam_result["threshold"],
            fpr_target=float(thresholds.get("suspicious_fpr_target", 0.05)),
            logger=log,
        )
        spam_t = float(spam_result["threshold"])
        suspicious_t = float(suspicious_result["threshold"])
        thresholds.update({
            "spam_threshold": round(spam_t, 6),
            "suspicious_threshold": round(suspicious_t, 6),
            "val_tune_metrics": spam_result["metrics"],
            "suspicious_metrics": suspicious_result["metrics"],
        })
        with open(artifacts_dir / "threshold_config.json", "w", encoding="utf-8") as fh:
            json.dump(thresholds, fh, ensure_ascii=False, indent=2)
        predictor.thresholds = thresholds
        log(f"Retuned: spam={spam_t:.4f}, suspicious={suspicious_t:.4f}")

    log("Computing metrics...")
    test_metrics = metrics_at_threshold(test_df["label"].to_numpy(dtype=int), test_probs, spam_t)
    real_metrics = metrics_at_threshold(test_real_df["label"].to_numpy(dtype=int), real_probs, spam_t)
    test_language = per_group(test_df, test_probs, spam_t, "language_group")
    test_category = per_group(test_df, test_probs, spam_t, "category_group")
    real_language = per_group(test_real_df, real_probs, spam_t, "language_group")
    real_category = per_group(test_real_df, real_probs, spam_t, "category_group")

    suite_path = resolve_path("tests") / "adversarial_suite.json"
    adversarial = {"total": 0, "hard_pass_rate": 0.0, "soft_pass_rate": 0.0,
                   "block_safe_rate": 0.0, "failures": [], "all_cases": [],
                   "false_blocks": 0, "false_reviews": 0, "missed_scams": 0}
    if suite_path.exists():
        log("Running adversarial suite...")
        suite = load_adversarial(suite_path)
        adversarial = adversarial_report(predictor, suite, spam_t, suspicious_t)
        log(
            f"Adversarial: hard={adversarial['hard_pass_rate']:.2%}, "
            f"soft={adversarial['soft_pass_rate']:.2%}, "
            f"block-safe={adversarial['block_safe_rate']:.2%}"
        )
    else:
        log(f"Adversarial suite not found at {suite_path}")

    plots = {}
    if not args.skip_plots:
        log("Generating plots...")
        plots = save_plots(
            test_real_df["label"].to_numpy(dtype=int),
            real_probs,
            spam_t,
            real_language,
            plots_dir,
        )

    metadata = predictor.metadata
    evaluation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "spam_threshold": spam_t,
            "suspicious_threshold": suspicious_t,
            "fpr_target": thresholds.get("fpr_target"),
        },
        "test": {
            "rows": int(len(test_df)),
            "prior": round(float(test_df["label"].mean()), 4),
            "metrics": test_metrics,
            "per_language": test_language,
            "per_category": test_category,
        },
        "test_realistic": {
            "rows": int(len(test_real_df)),
            "prior": round(float(test_real_df["label"].mean()), 4),
            "metrics": real_metrics,
            "per_language": real_language,
            "per_category": real_category,
        },
        "adversarial": adversarial,
        "plots": plots,
        "dataset": {
            "test_sources": {
                str(k): int(v) for k, v in test_df["source"].value_counts().head(20).items()
            },
        },
    }
    with open(artifacts_dir / "evaluation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, ensure_ascii=False, indent=2, default=json_default)
    with open(artifacts_dir / "adversarial_report.json", "w", encoding="utf-8") as fh:
        json.dump(adversarial, fh, ensure_ascii=False, indent=2, default=json_default)
    write_report(
        artifacts_dir / "evaluation_report.md",
        evaluation,
        metadata,
        adversarial,
        thresholds,
    )

    log("Held-out realistic test:")
    for key in ("pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "ece", "brier", "accuracy"):
        log(f"  {key}: {real_metrics.get(key)}")
    log("Full held-out test:")
    for key in ("pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "ece", "brier", "accuracy"):
        log(f"  {key}: {test_metrics.get(key)}")
    log(f"Wrote report to {artifacts_dir / 'evaluation_report.md'}")
    if adversarial.get("failures"):
        log(f"Adversarial failures: {len(adversarial['failures'])}")
        for failure in adversarial["failures"][:10]:
            log(
                f"  {failure['expected']} -> {failure['predicted']} "
                f"p={failure['spam_probability']:.3f} | {failure['text'][:120]}"
            )


if __name__ == "__main__":
    main()
