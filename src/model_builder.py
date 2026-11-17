from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .features import (
    FEATURE_NAMES,
    SignalFeatures,
    feature_vector_cached,
    prepare_for_model,
)
from .utils import ensure_dir, resolve_path, set_seed

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
HEURISTIC_INDEX = FEATURE_INDEX["heuristic_score"]


class HybridFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        char_ngram_range=(2, 5),
        char_max_features=180000,
        word_ngram_range=(1, 2),
        word_max_features=110000,
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
    ):
        self.char_ngram_range = tuple(char_ngram_range)
        self.char_max_features = int(char_max_features)
        self.word_ngram_range = tuple(word_ngram_range)
        self.word_max_features = int(word_max_features)
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = bool(sublinear_tf)

    def fit(self, X, y=None):
        texts = [prepare_for_model(t) for t in X]
        self.char_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            max_features=self.char_max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=False,
            dtype=np.float32,
        )
        self.word_ = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.word_ngram_range,
            max_features=self.word_max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        self.char_.fit(texts)
        self.word_.fit(texts)
        self.signal_ = SignalFeatures().fit(X)
        self.n_char_ = len(self.char_.get_feature_names_out())
        self.n_word_ = len(self.word_.get_feature_names_out())
        self.n_signal_ = len(FEATURE_NAMES)
        return self

    def transform(self, X):
        texts = [prepare_for_model(t) for t in X]
        char = self.char_.transform(texts)
        word = self.word_.transform(texts)
        signal = self.signal_.transform(X)
        return sp.hstack([char, word, signal], format="csr").astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        char = [f"char:{x}" for x in self.char_.get_feature_names_out()]
        word = [f"word:{x}" for x in self.word_.get_feature_names_out()]
        signal = [f"signal:{x}" for x in FEATURE_NAMES]
        return np.asarray(char + word + signal, dtype=object)

    @property
    def n_features_(self):
        return int(self.n_char_ + self.n_word_ + self.n_signal_)

    def explain(self, text: str, coef: np.ndarray, top_k: int = 8) -> dict:
        x = self.transform([text]).tocsr()
        row = x.getrow(0)
        if getattr(self, "_char_names_", None) is None:
            self._char_names_ = self.char_.get_feature_names_out()
            self._word_names_ = self.word_.get_feature_names_out()
        char_names = self._char_names_
        word_names = self._word_names_
        cues = []
        for idx, value in zip(row.indices, row.data):
            idx = int(idx)
            if idx < self.n_char_:
                name = "char:" + str(char_names[idx])
                kind = "char"
            elif idx < self.n_char_ + self.n_word_:
                name = "word:" + str(word_names[idx - self.n_char_])
                kind = "word"
            else:
                name = "signal:" + FEATURE_NAMES[idx - self.n_char_ - self.n_word_]
                kind = "signal"
            contribution = float(value * coef[idx])
            cues.append({"feature": name, "type": kind, "contribution": contribution})
        positive = [c for c in cues if c["contribution"] > 0]
        negative = [c for c in cues if c["contribution"] < 0]
        word_pos = sorted([c for c in positive if c["type"] != "char"], key=lambda c: c["contribution"], reverse=True)
        char_pos = sorted([c for c in positive if c["type"] == "char"], key=lambda c: c["contribution"], reverse=True)
        risk = word_pos[:top_k]
        if len(risk) < top_k:
            risk += char_pos[: top_k - len(risk)]
        protective = sorted([c for c in negative if c["type"] != "char"], key=lambda c: c["contribution"])[:top_k]
        return {
            "risk_cues": [
                {"feature": c["feature"], "type": c["type"], "contribution": round(c["contribution"], 5)}
                for c in risk
            ],
            "protective_cues": [
                {"feature": c["feature"], "type": c["type"], "contribution": round(c["contribution"], 5)}
                for c in protective
            ],
        }


def stratified_indices(y: np.ndarray, n: int, seed: int, strata: Sequence | None = None) -> np.ndarray:
    y = np.asarray(y)
    idx = np.arange(len(y))
    if n >= len(y):
        return idx
    labels = y.astype(str)
    if strata is not None:
        labels = np.char.add(np.char.add(labels.astype(str), "|"), np.asarray(strata).astype(str))
    try:
        chosen, _ = train_test_split(
            idx, train_size=int(n), stratify=labels, random_state=seed
        )
        return np.sort(chosen)
    except ValueError:
        rng = np.random.RandomState(seed)
        return np.sort(rng.choice(idx, size=int(n), replace=False))


def train_lexical_models(
    train_df,
    val_df,
    config: dict,
    logger=print,
) -> dict:
    cfg = config["model"]["lexical"]
    seed = int(config["project"]["seed"])
    set_seed(seed)
    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)
    X_train = train_df["text"].astype(str).tolist()
    X_val = val_df["text"].astype(str).tolist()

    feature = HybridFeatureExtractor(
        char_ngram_range=cfg.get("char_ngram_range", (2, 5)),
        char_max_features=cfg.get("char_max_features", 180000),
        word_ngram_range=cfg.get("word_ngram_range", (1, 2)),
        word_max_features=cfg.get("word_max_features", 110000),
        min_df=cfg.get("min_df", 2),
        max_df=cfg.get("max_df", 0.995),
        sublinear_tf=cfg.get("sublinear_tf", True),
    )
    t0 = time.time()
    logger("Fitting hybrid feature extractor...")
    feature.fit(X_train)
    logger(
        f"Feature fit done in {time.time() - t0:.1f}s "
        f"(char={feature.n_char_:,}, word={feature.n_word_:,}, signal={feature.n_signal_})"
    )
    t0 = time.time()
    Xtr = feature.transform(X_train)
    Xva = feature.transform(X_val)
    logger(f"Feature transforms done in {time.time() - t0:.1f}s, nnz(train)={Xtr.nnz:,}")

    grid_n = min(len(y_train), 40000)
    grid_idx = stratified_indices(
        y_train, grid_n, seed, strata=train_df["language"].to_numpy()
    )
    X_grid = Xtr[grid_idx]
    y_grid = y_train[grid_idx]
    logger(f"Tuning lexical C on {len(grid_idx):,} language-stratified rows")

    grid_results = []
    fitted = {}
    for C in cfg.get("C_grid", [1.0]):
        t0 = time.time()
        model = LogisticRegression(
            C=float(C),
            class_weight=cfg.get("class_weight", "balanced"),
            solver="liblinear",
            max_iter=int(cfg.get("max_iter", 400)),
            random_state=seed,
        )
        model.fit(X_grid, y_grid)
        proba = model.predict_proba(Xva)[:, 1]
        pr_auc = float(average_precision_score(y_val, proba))
        f1 = float(f1_score(y_val, (proba >= 0.5).astype(int), zero_division=0))
        grid_results.append({"C": float(C), "pr_auc": round(pr_auc, 5), "f1": round(f1, 5)})
        fitted[float(C)] = model
        logger(f"  C={C}: PR-AUC={pr_auc:.4f}, F1@0.5={f1:.4f} ({time.time() - t0:.1f}s)")

    best = max(grid_results, key=lambda r: (r["pr_auc"], r["f1"]))
    best_C = best["C"]
    logger(f"Best lexical C={best_C}; refitting on all {len(y_train):,} rows")
    t0 = time.time()
    lr = LogisticRegression(
        C=best_C,
        class_weight=cfg.get("class_weight", "balanced"),
        solver="liblinear",
        max_iter=int(cfg.get("max_iter", 400)),
        random_state=seed,
    )
    lr.fit(Xtr, y_train)
    logger(f"Final logistic regression fit in {time.time() - t0:.1f}s")

    t0 = time.time()
    svm = LinearSVC(
        C=float(cfg.get("svm_C", 1.0)),
        class_weight=cfg.get("class_weight", "balanced"),
        max_iter=3000,
        dual="auto",
        random_state=seed,
    )
    svm.fit(Xtr, y_train)
    logger(f"LinearSVC fit in {time.time() - t0:.1f}s")

    lr_proba = lr.predict_proba(Xva)[:, 1]
    svm_score = svm.decision_function(Xva)
    val_metrics = {
        "logistic": {
            "pr_auc": round(float(average_precision_score(y_val, lr_proba)), 5),
            "f1_at_0_5": round(float(f1_score(y_val, (lr_proba >= 0.5).astype(int), zero_division=0)), 5),
        },
        "svm": {
            "pr_auc": round(float(average_precision_score(y_val, svm_score)), 5),
            "f1_at_0": round(float(f1_score(y_val, (svm_score >= 0).astype(int), zero_division=0)), 5),
        },
    }
    logger(f"Lexical validation: {json.dumps(val_metrics, ensure_ascii=False)}")
    return {
        "feature_extractor": feature,
        "lr": lr,
        "svm": svm,
        "best_C": best_C,
        "svm_C": float(cfg.get("svm_C", 1.0)),
        "grid_results": grid_results,
        "val_metrics": val_metrics,
        "feature_count": feature.n_features_,
        "train_rows": int(len(y_train)),
    }


class SemanticEncoder:
    def __init__(
        self,
        model_name: str,
        onnx_path: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
        max_seq_length: int = 128,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.onnx_path = str(onnx_path) if onnx_path else None
        self.tokenizer_dir = str(tokenizer_dir) if tokenizer_dir else None
        self.max_seq_length = int(max_seq_length)
        self.batch_size = int(batch_size)
        self.device = device
        self._loaded = False
        self._session = None
        self._tokenizer = None
        self._st = None

    def _load(self):
        if self._loaded:
            return
        if self.onnx_path and Path(self.onnx_path).exists():
            import onnxruntime as ort
            from transformers import AutoTokenizer

            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(self.onnx_path, providers=providers)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_dir or self.model_name
            )
        else:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
            self._st = SentenceTransformer(self.model_name, device=self.device)
            self._st.max_seq_length = self.max_seq_length
        self._loaded = True

    def _encode_onnx(self, texts: list[str], batch_size: int) -> np.ndarray:
        input_names = {i.name for i in self._session.get_inputs()}
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            feed = {}
            for name in ("input_ids", "attention_mask", "token_type_ids"):
                if name in input_names and name in enc:
                    feed[name] = enc[name].astype(np.int64)
            result = self._session.run(None, feed)[0]
            if result.ndim == 3:
                mask = enc["attention_mask"].astype(np.float32)
                mask = np.expand_dims(mask, -1)
                result = (result * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            norm = np.linalg.norm(result, axis=1, keepdims=True)
            outputs.append(result / np.clip(norm, 1e-9, None))
        return np.vstack(outputs).astype(np.float32) if outputs else np.empty((0, 384), dtype=np.float32)

    def encode(self, texts: Sequence[str], batch_size: int | None = None, show_progress: bool = False) -> np.ndarray:
        self._load()
        items = [str(t) for t in texts]
        if not items:
            return np.empty((0, 384), dtype=np.float32)
        bs = int(batch_size or self.batch_size)
        if self._session is not None:
            return self._encode_onnx(items, bs)
        return np.asarray(
            self._st.encode(
                items,
                batch_size=bs,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            ),
            dtype=np.float32,
        )

    def encode_to_file(
        self,
        texts: Sequence[str],
        path: str | Path,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        path = Path(path)
        manifest_path = path.with_suffix(path.suffix + ".json")
        items = [str(t) for t in texts]
        self._load()
        backend = "onnx" if self._session is not None else "pytorch"
        manifest = {
            "model": self.model_name,
            "backend": backend,
            "count": len(items),
            "max_seq_length": self.max_seq_length,
            "digest": hashlib.blake2b(
                ("\n".join(items)).encode("utf-8"), digest_size=16
            ).hexdigest(),
        }
        if path.exists() and manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    old = json.load(fh)
                if old == manifest:
                    return np.load(path)
            except Exception:
                pass
        embeddings = self.encode(items, batch_size=batch_size, show_progress=show_progress)
        np.save(path, embeddings)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        return embeddings


def train_semantic_head(
    embeddings_train: np.ndarray,
    y_train: np.ndarray,
    embeddings_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    logger=print,
) -> dict:
    cfg = config["model"]["semantic"]
    seed = int(config["project"]["seed"])
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(embeddings_train)
    Xva = scaler.transform(embeddings_val)
    results = []
    models = {}
    for C in cfg.get("C_grid", [1.0]):
        model = LogisticRegression(
            C=float(C),
            class_weight=cfg.get("class_weight", "balanced"),
            max_iter=2000,
            random_state=seed,
        )
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xva)[:, 1]
        pr_auc = float(average_precision_score(y_val, proba))
        f1 = float(f1_score(y_val, (proba >= 0.5).astype(int), zero_division=0))
        results.append({"C": float(C), "pr_auc": round(pr_auc, 5), "f1": round(f1, 5)})
        models[float(C)] = model
        logger(f"  semantic C={C}: PR-AUC={pr_auc:.4f}, F1@0.5={f1:.4f}")
    best = max(results, key=lambda r: (r["pr_auc"], r["f1"]))
    head = models[best["C"]]
    return {
        "head": head,
        "scaler": scaler,
        "best_C": best["C"],
        "grid_results": results,
        "val_metrics": best,
        "embedding_dim": int(embeddings_train.shape[1]),
    }


def _logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def calibrator_columns(
    lr_probs: np.ndarray,
    svm_scores: np.ndarray,
    sem_probs: np.ndarray | None,
    heuristic_scores: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    cols = [_logit(lr_probs), np.asarray(svm_scores, dtype=np.float64)]
    names = ["lexical_lr_logit", "lexical_svm_score"]
    if sem_probs is not None:
        cols.append(_logit(sem_probs))
        names.append("semantic_logit")
    cols.append(np.asarray(heuristic_scores, dtype=np.float64))
    names.append("heuristic_score")
    return np.column_stack(cols), names


def fit_calibrator(
    lr_probs: np.ndarray,
    svm_scores: np.ndarray,
    sem_probs: np.ndarray | None,
    heuristic_scores: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    logger=print,
) -> dict:
    X, names = calibrator_columns(lr_probs, svm_scores, sem_probs, heuristic_scores)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    stacker.fit(Xs, y)
    proba = stacker.predict_proba(Xs)[:, 1]
    logger(f"Calibrator fit on {len(y):,} rows; in-sample PR-AUC={average_precision_score(y, proba):.4f}")
    return {
        "scaler": scaler,
        "stacker": stacker,
        "feature_names": names,
        "train_rows": int(len(y)),
        "train_prior": float(np.mean(y)),
    }


def apply_calibrator(
    calibrator: dict,
    lr_probs: np.ndarray,
    svm_scores: np.ndarray,
    sem_probs: np.ndarray | None,
    heuristic_scores: np.ndarray,
) -> np.ndarray:
    X, _ = calibrator_columns(lr_probs, svm_scores, sem_probs, heuristic_scores)
    return calibrator["stacker"].predict_proba(calibrator["scaler"].transform(X))[:, 1]


def heuristic_scores(texts: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [feature_vector_cached(str(t))[HEURISTIC_INDEX] for t in texts], dtype=np.float64
    )


def predict_lexical(lexical: dict, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    X = lexical["feature_extractor"].transform([str(t) for t in texts])
    lr_probs = lexical["lr"].predict_proba(X)[:, 1]
    svm_scores = lexical["svm"].decision_function(X)
    return lr_probs, svm_scores, X


def predict_semantic(semantic: dict, texts: Sequence[str], show_progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
    embeddings = semantic["encoder"].encode(texts, show_progress=show_progress)
    probs = semantic["head"].predict_proba(semantic["scaler"].transform(embeddings))[:, 1]
    return embeddings, probs


def predict_final(
    lexical: dict,
    semantic: dict | None,
    calibrator: dict,
    texts: Sequence[str],
    show_progress: bool = False,
) -> dict:
    lr_probs, svm_scores, _ = predict_lexical(lexical, texts)
    sem_probs = None
    if semantic is not None:
        _, sem_probs = predict_semantic(semantic, texts, show_progress=show_progress)
    h = heuristic_scores(texts)
    final = apply_calibrator(calibrator, lr_probs, svm_scores, sem_probs, h)
    return {
        "probability": final,
        "lexical_lr": lr_probs,
        "lexical_svm": svm_scores,
        "semantic": sem_probs,
        "heuristic": h,
    }


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "fpr": round(float(fpr), 6),
        "fnr": round(float(fnr), 6),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "predicted_spam": int(pred.sum()),
    }


def tune_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    fpr_target: float = 0.01,
    logger=print,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    candidates = np.unique(
        np.round(np.concatenate([np.linspace(0.005, 0.995, 199), probs]), 6)
    )
    best = None
    for threshold in candidates:
        metrics = threshold_metrics(y_true, probs, float(threshold))
        if metrics["fpr"] <= fpr_target + 1e-12:
            if best is None or (metrics["recall"], metrics["precision"]) > (
                best["metrics"]["recall"], best["metrics"]["precision"]
            ):
                best = {"threshold": float(threshold), "metrics": metrics}
    if best is None:
        fallback = min(
            (threshold_metrics(y_true, probs, float(t)) for t in candidates),
            key=lambda m: (m["fpr"], -m["recall"]),
        )
        best = {"threshold": fallback["threshold"], "metrics": fallback}
    logger(
        f"Threshold {best['threshold']:.4f} | FPR={best['metrics']['fpr']:.4f} "
        f"Recall={best['metrics']['recall']:.4f} Precision={best['metrics']['precision']:.4f} "
        f"(target FPR <= {fpr_target:.2%})"
    )
    return {
        "threshold": best["threshold"],
        "metrics": best["metrics"],
        "fpr_target": float(fpr_target),
    }


def suspicious_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    spam_threshold_value: float,
    fpr_target: float = 0.05,
    logger=print,
) -> dict:
    result = tune_threshold(y_true, probs, fpr_target=fpr_target, logger=logger)
    threshold = float(result["threshold"])
    if threshold >= spam_threshold_value:
        threshold = max(0.01, min(spam_threshold_value - 0.001, spam_threshold_value * 0.5))
    return {
        "threshold": round(float(threshold), 6),
        "metrics": result["metrics"],
        "fpr_target": float(fpr_target),
    }


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
