from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import analyze_text, detect_language
from .synthetic import generate_synthetic
from .utils import (
    PROJECT_ROOT,
    canonical_for_dedupe,
    clean_text,
    dedupe_simhash,
    ensure_dir,
    extract_urls,
    is_legitimate_domain,
    load_config,
    resolve_path,
    set_seed,
    text_hash,
)

RAW_COLUMNS = [
    "text", "label", "language", "category", "source", "group",
    "trust", "raw_id",
]

TRUST_PRIORITY = {"gold": 0, "silver": 1, "noisy": 2, "synthetic": 3}

SOURCE_INFO = {
    "scamshield": {
        "title": "ScamShield Multilingual Smishing & Social Engineering",
        "license": "MIT",
        "note": "Real and synthetic SMS covering English, Hindi, Hinglish, UPI/KYC/courier scams. Original splits were discarded and global leakage-safe splits were rebuilt.",
    },
    "uci_sms": {
        "title": "UCI SMS Spam Collection",
        "license": "Unknown / research use",
        "note": "Classic 5,572-message English SMS spam corpus.",
    },
    "bengali_smishing": {
        "title": "Bengali SMS Smishing Dataset",
        "license": "MIT",
        "note": "Bengali, Banglish, English and code-mixed SMS labelled smish/normal/promo.",
    },
    "hindi_sms": {
        "title": "Hindi SMS Spam-Ham 15K",
        "license": "Unknown",
        "note": "Balanced Hindi/Hinglish SMS corpus.",
    },
    "arabic_sms": {
        "title": "Arabic SMS Spam Dataset",
        "license": "GPL-3.0",
        "note": "Arabic ham/spam SMS.",
    },
    "spanish_sms": {
        "title": "SMS Spam Mexico",
        "license": "CC-BY-SA-4.0",
        "note": "Mexican Spanish SMS phishing and ham examples.",
    },
    "french_reviews": {
        "title": "French Spam/Ham Detection Free 2K",
        "license": "Other",
        "note": "French promotional/phishing and ham texts. Used as a secondary multilingual source.",
    },
    "merged_sms": {
        "title": "SMS Scam Detection Dataset Merged",
        "license": "Other",
        "note": "Large merged SMS scam/ham corpus with partial language labels.",
    },
    "multilang_sms": {
        "title": "MultiLanguage SMS Classification Dataset",
        "license": "Unknown",
        "note": "Promotional vs service SMS. Service messages are valuable hard negatives (OTP, transaction alerts, delivery updates).",
    },
    "p2p_sms": {
        "title": "A2P vs P2P SMS Classifier",
        "license": "MIT",
        "note": "Peer-to-peer and application-to-person SMS with Hinglish examples. Used for hard negatives and promotional positives.",
    },
    "phishing_email": {
        "title": "Phishing Email & SMS with NLP Categories",
        "license": "MIT",
        "note": "English phishing positives grouped by urgency, authority and persuasion.",
    },
    "synthetic": {
        "title": "Template-based multilingual augmentation",
        "license": "Generated for this project",
        "note": "Modern P2P, crypto, marketplace, delivery/OTP, romance, job and bank scams plus hard negatives across English, Bengali, Banglish, Hindi, Hinglish, Spanish, French, Arabic and code-mixed text.",
    },
}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_COLUMNS)


def _canonical_language(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    v = value.strip().casefold()
    mapping = {
        "en": "English", "eng": "English", "english": "English",
        "hi": "Hindi", "hin": "Hindi", "hindi": "Hindi",
        "hinglish": "Hinglish", "roman hindi": "Hinglish",
        "bn": "Bengali", "ben": "Bengali", "bengali": "Bengali",
        "bangla": "Bengali", "banglish": "Banglish",
        "es": "Spanish", "spa": "Spanish", "spanish": "Spanish",
        "fr": "French", "fra": "French", "fre": "French", "french": "French",
        "ar": "Arabic", "ara": "Arabic", "arabic": "Arabic",
        "de": "German", "german": "German", "ru": "Russian",
        "russian": "Russian", "zh": "Chinese", "chinese": "Chinese",
        "mixed": "mixed", "code-mix": "mixed", "codemix": "mixed",
        "code_mix": "mixed", "und": None, "unknown": None,
    }
    return mapping.get(v, value.strip() if len(value.strip()) <= 24 else "other")


def _detect(text: str, fallback: str = "other") -> str:
    try:
        lang, _, _ = detect_language(text)
        if not lang or lang == "unknown":
            return fallback
        return lang
    except Exception:
        return fallback


def _hindi_language(text: str) -> str:
    try:
        lang, _, shares = detect_language(text)
    except Exception:
        return "Hindi"
    if shares.get("Devanagari", 0.0) > 0.2:
        return "Hindi"
    if lang in ("Hinglish", "mixed"):
        return lang
    return "Hindi"


def _finalize(
    df: pd.DataFrame,
    source: str,
    trust: str,
    language_fn: Callable[[pd.Series], pd.Series] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return _empty()
    out = df.copy()
    out["text"] = out["text"].map(clean_text)
    out = out[out["text"].str.len() > 0].copy()
    out["label"] = out["label"].astype(int)
    out["source"] = (
        source.reindex(out.index) if isinstance(source, pd.Series) else source
    )
    out["trust"] = (
        trust.reindex(out.index) if isinstance(trust, pd.Series) else trust
    )
    if "language" not in out:
        out["language"] = None
    if language_fn is not None:
        out["language"] = language_fn(out)
    out["language"] = out["language"].map(_canonical_language)
    out["language"] = out["language"].fillna("other")
    out["category"] = out["category"].fillna("unknown").astype(str)
    out["raw_id"] = out["raw_id"].astype(str)
    out["group"] = out["group"].astype(str)
    for col in RAW_COLUMNS:
        if col not in out:
            out[col] = ""
    return out[RAW_COLUMNS]


def load_scamshield(config: dict) -> pd.DataFrame:
    base = resolve_path(config["paths"]["raw_dir"]) / "scamshield"
    frames = []
    for split in ("train", "val", "test"):
        path = base / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame["source_dataset"] = frame.get("source_dataset", "").fillna("")
        frame = frame[frame["source_dataset"] != "Kaggle_Hindi_Merged"]
        frames.append(frame)
    if not frames:
        return _empty()
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("")
    df["label"] = df["is_scam"].fillna(0).astype(int)
    lang_map = {"English": "English", "Hindi": "Hindi", "Hinglish": "Hinglish"}
    df["language"] = df["language"].map(lang_map).fillna("other")
    df["category"] = (
        df["head2_scam_intent"].fillna("Legitimate / Benign")
        .str.casefold().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    )
    df.loc[df["label"] == 0, "category"] = "benign"
    df["source"] = "scamshield:" + df["source_dataset"].astype(str)
    df["trust"] = np.select(
        [
            df["source_dataset"].isin(["UCI_SMS_Spam", "Benchmark_Ground_Truth", "ysangam/Indian_Cyber_Scam_Hinglish"]),
            df["source_dataset"].eq("Synthetic_Tier_C"),
        ],
        ["gold", "silver"],
        default="noisy",
    )
    df["group"] = "scamshield:" + df["source_dataset"].astype(str) + ":" + df["id"].astype(str)
    df["raw_id"] = "scamshield:" + df["id"].astype(str)
    return _finalize(df, df["source"], df["trust"])


def load_uci(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["raw_dir"]) / "uci" / "spam.csv"
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, encoding="latin-1")
    if "v1" not in df or "v2" not in df:
        return _empty()
    out = pd.DataFrame({
        "text": df["v2"].fillna(""),
        "label": df["v1"].astype(str).str.casefold().eq("spam").astype(int),
        "category": np.where(df["v1"].astype(str).str.casefold().eq("spam"), "sms_spam", "benign"),
        "group": [f"uci:{i}" for i in range(len(df))],
        "raw_id": [f"uci:{i}" for i in range(len(df))],
    })
    out["language"] = "English"
    return _finalize(out, "uci_sms", "gold")


def load_bengali(config: dict) -> pd.DataFrame:
    base = resolve_path(config["paths"]["raw_dir"]) / "bengali" / "bengali-sms-smishing-dataset" / "data"
    frames = []
    for split in ("train", "validation", "test"):
        path = base / f"{split}-00000-of-00001.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return _empty()
    df = pd.concat(frames, ignore_index=True)
    label_map = {"smish": 1, "promo": 1, "normal": 0}
    raw_label = df["label"].astype(str).str.casefold()
    df["label"] = raw_label.map(label_map).fillna(0).astype(int)
    df["category"] = raw_label.map(
        {"smish": "bengali_smishing", "promo": "bengali_promotional", "normal": "benign"}
    ).fillna("benign")
    source_map = {
        "Banglish": "Banglish", "Bengali": "Bengali", "English": "English",
        "CodeMix": "mixed",
    }
    df["language"] = df["source"].map(source_map).fillna("Bengali")
    df["group"] = [f"bengali:{i}" for i in range(len(df))]
    df["raw_id"] = [f"bengali:{i}" for i in range(len(df))]
    df["source"] = "bengali_smishing"
    df["trust"] = "gold"
    return _finalize(df, "bengali_smishing", "gold")


def load_hindi(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["raw_dir"]) / "hindi" / "hindi_sms_spam_ham_15000.csv"
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "sms" not in df or "label" not in df:
        return _empty()
    raw_label = df["label"].astype(str).str.casefold()
    out = pd.DataFrame({
        "text": df["sms"].fillna(""),
        "label": raw_label.eq("spam").astype(int),
        "category": raw_label.map({"spam": "hindi_sms_spam", "ham": "benign"}).fillna("benign"),
        "group": [f"hindi:{i}" for i in range(len(df))],
        "raw_id": [f"hindi:{i}" for i in range(len(df))],
    })
    out["language"] = out["text"].map(_hindi_language)
    return _finalize(out, "hindi_sms", "gold")


def load_arabic(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["raw_dir"]) / "arabic" / "arabic_sms_spam.csv"
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "Message" not in df or "Label" not in df:
        return _empty()
    out = pd.DataFrame({
        "text": df["Message"].fillna(""),
        "label": pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int),
        "language": "Arabic",
        "category": np.where(pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int) == 1,
                             "arabic_sms_spam", "benign"),
        "group": [f"arabic:{i}" for i in range(len(df))],
        "raw_id": [f"arabic:{i}" for i in range(len(df))],
    })
    return _finalize(out, "arabic_sms", "gold")


def load_spanish(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["raw_dir"]) / "spanish" / "dataset.csv"
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "texto" not in df or "etiqueta" not in df:
        return _empty()
    raw_label = df["etiqueta"].astype(str).str.casefold()
    out = pd.DataFrame({
        "text": df["texto"].fillna(""),
        "label": raw_label.eq("spam").astype(int),
        "language": "Spanish",
        "category": raw_label.map(
            {"spam": "spanish_sms_spam", "ham": "benign"}
        ).fillna("benign"),
        "group": [f"spanish:{i}" for i in range(len(df))],
        "raw_id": [f"spanish:{i}" for i in range(len(df))],
    })
    return _finalize(out, "spanish_sms", "gold")


def load_french(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["raw_dir"]) / "french" / "data.jsonl"
    if not path.exists():
        return _empty()
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    if "text" not in df or "label" not in df:
        return _empty()
    raw_label = df["label"].astype(str).str.casefold()
    out = pd.DataFrame({
        "text": df["text"].fillna(""),
        "label": raw_label.eq("spam").astype(int),
        "language": "French",
        "category": raw_label.map(
            {"spam": "french_spam", "ham": "benign"}
        ).fillna("benign"),
        "group": [f"french:{i}" for i in range(len(df))],
        "raw_id": [f"french:{i}" for i in range(len(df))],
    })
    return _finalize(out, "french_reviews", "silver")


def load_merged(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["raw_dir"]) / "merged" / "sms_scam_detection_dataset_merged_with_lang.csv"
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    if "text" not in df or "label" not in df:
        return _empty()
    df["label_norm"] = df["label"].astype(str).str.casefold().str.strip()
    df = df[df["label_norm"].isin(["ham", "spam"])].copy()
    df["label"] = (df["label_norm"] == "spam").astype(int)
    lang_series = df["lang"] if "lang" in df else pd.Series([None] * len(df))
    provided = lang_series.map(_canonical_language)
    detected = df["text"].map(lambda t: _detect(t, "English"))
    df["language"] = provided.where(provided.notna(), detected)
    out = pd.DataFrame({
        "text": df["text"].fillna(""),
        "label": df["label"],
        "language": df["language"],
        "category": np.where(df["label"] == 1, "merged_sms_spam", "benign"),
        "group": [f"merged:{i}" for i in range(len(df))],
        "raw_id": [f"merged:{i}" for i in range(len(df))],
    })
    return _finalize(out, "merged_sms", "silver")


def load_multilang(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["raw_dir"]) / "multilang" / "FINAL_MERGED_SMS_DATASET.csv"
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    if "message" not in df or "category" not in df:
        return _empty()
    df["category_norm"] = df["category"].astype(str).str.casefold().str.strip()
    df = df[df["category_norm"].isin(["promotional", "service"])].copy()
    df["label"] = (df["category_norm"] == "promotional").astype(int)
    out = pd.DataFrame({
        "text": df["message"].fillna(""),
        "label": df["label"],
        "language": df["message"].map(lambda t: _detect(t, "English")),
        "category": np.where(df["label"] == 1, "promotional_sms", "service_sms"),
        "group": [f"multilang:{i}" for i in range(len(df))],
        "raw_id": [f"multilang:{i}" for i in range(len(df))],
    })
    return _finalize(out, "multilang_sms", "silver")


def load_p2p(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["raw_dir"]) / "p2p" / "a2p_p2p_sms_dataset.csv"
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    if "message_text" not in df or "label" not in df:
        return _empty()
    label_norm = df["label"].astype(str).str.casefold().str.strip()
    sub = df.get("sub_category", pd.Series(["unknown"] * len(df))).astype(str)
    is_promotional = label_norm.eq("a2p") & sub.str.casefold().str.contains("promotional", na=False)
    df["label"] = is_promotional.astype(int)
    hinglish = pd.to_numeric(df.get("is_hinglish", 0), errors="coerce").fillna(0).astype(int).eq(1)
    detected = df["message_text"].map(lambda t: _detect(t, "English"))
    df["language"] = np.where(hinglish, "Hinglish", detected)
    df["category"] = sub.str.casefold().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    df.loc[df["label"] == 0, "category"] = "benign_" + df.loc[df["label"] == 0, "category"]
    out = pd.DataFrame({
        "text": df["message_text"].fillna(""),
        "label": df["label"],
        "language": df["language"],
        "category": df["category"],
        "group": [f"p2p:{i}" for i in range(len(df))],
        "raw_id": [f"p2p:{i}" for i in range(len(df))],
    })
    return _finalize(out, "p2p_sms", "silver")


def load_urgency(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["raw_dir"]) / "phishing" / "urgency" / "phishing_dataset_with_category.csv"
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in df:
        return _empty()
    category = df.get("category", pd.Series(["phishing"] * len(df))).astype(str).str.casefold()
    out = pd.DataFrame({
        "text": df["text"].fillna(""),
        "label": 1,
        "language": "English",
        "category": "phishing_" + category,
        "group": [f"phishing:{i}" for i in range(len(df))],
        "raw_id": [f"phishing:{i}" for i in range(len(df))],
    })
    return _finalize(out, "phishing_email", "silver")


def load_synthetic(config: dict) -> pd.DataFrame:
    rows = generate_synthetic(config, seed=int(config["project"]["seed"]))
    if not rows:
        return _empty()
    df = pd.DataFrame([r.__dict__ for r in rows])
    df["raw_id"] = [f"synthetic:{i}" for i in range(len(df))]
    return _finalize(df, "synthetic", "synthetic")


LOADERS: list[tuple[str, Callable[[dict], pd.DataFrame], str]] = [
    ("scamshield", load_scamshield, "scamshield"),
    ("uci", load_uci, "uci"),
    ("bengali", load_bengali, "bengali"),
    ("hindi", load_hindi, "hindi"),
    ("arabic", load_arabic, "arabic"),
    ("spanish", load_spanish, "spanish"),
    ("french", load_french, "french"),
    ("merged", load_merged, "merged"),
    ("multilang", load_multilang, "multilang"),
    ("p2p", load_p2p, "p2p"),
    ("urgency", load_urgency, "urgency"),
]


def _cap_source(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0 or len(df) <= cap:
        return df
    df = df.copy()
    df["_strat"] = df["label"].astype(str) + "|" + df["language"].astype(str)
    frac = cap / len(df)
    pieces = []
    for _, group in df.groupby("_strat", sort=False):
        n = max(1, int(round(len(group) * frac)))
        pieces.append(group.sample(min(n, len(group)), random_state=seed))
    sampled = pd.concat(pieces, ignore_index=False)
    if len(sampled) > cap:
        sampled = sampled.sample(cap, random_state=seed)
    elif len(sampled) < cap:
        remaining = df.drop(index=sampled.index, errors="ignore")
        sampled = pd.concat(
            [sampled, remaining.sample(min(cap - len(sampled), len(remaining)), random_state=seed)]
        )
    return sampled.drop(columns=["_strat"]).sort_index()


def _quality_filter(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    qcfg = config["data"]["quality"]
    stats = {"examined": 0, "dropped_negative_phishing": 0, "dropped_positive_legit": 0}
    if not qcfg.get("drop_ambiguous_negative_phishing") and not qcfg.get("drop_legitimate_domain_spam"):
        return df, stats
    mask = (df["trust"] == "noisy") & df["text"].str.contains(
        r"http|www|bank|account|otp|kyc|verify|suspend|block|prize|won|lottery|"
        r"bkash|nagad|paypal|upi|wallet|bitcoin|crypto",
        case=False, regex=True, na=False,
    )
    candidate_idx = df.index[mask]
    drop_idx: set = set()
    for i, idx in enumerate(candidate_idx):
        text = df.at[idx, "text"]
        label = int(df.at[idx, "label"])
        analysis = analyze_text(text)
        has_legit = any(is_legitimate_domain(u) for u in analysis.urls)
        has_suspicious = any(
            u.is_shortener or u.is_suspicious_tld or u.is_ip or u.is_obfuscated
            for u in analysis.urls
        )
        score = analysis.features.get("heuristic_score", 0.0)
        if (
            label == 0
            and qcfg.get("drop_ambiguous_negative_phishing")
            and score >= float(qcfg.get("ambiguous_negative_threshold", 0.78))
            and (has_suspicious or not analysis.urls)
        ):
            drop_idx.add(idx)
            stats["dropped_negative_phishing"] += 1
        elif (
            label == 1
            and qcfg.get("drop_legitimate_domain_spam")
            and score <= float(qcfg.get("legitimate_domain_spam_threshold", 0.18))
            and has_legit
            and not has_suspicious
        ):
            drop_idx.add(idx)
            stats["dropped_positive_legit"] += 1
        stats["examined"] += 1
        if (i + 1) % 10000 == 0:
            print(f"  quality filter {i + 1}/{len(candidate_idx)}", flush=True)
    if drop_idx:
        df = df.drop(index=list(drop_idx))
    return df, stats


def _dedupe(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    dcfg = config["data"]["dedupe"]
    print(f"  near-duplicate scan over {len(df):,} rows", flush=True)
    clusters = dedupe_simhash(
        df["text"].tolist(),
        hamming_threshold=int(dcfg.get("hamming_threshold", 3)),
        bands=int(dcfg.get("bands", 4)),
        show_progress=True,
    )
    work = df.copy()
    work["_cluster"] = clusters
    work["_trust_priority"] = work["trust"].map(TRUST_PRIORITY).fillna(9).astype(int)
    work["_text_len"] = work["text"].str.len()

    label_nunique = work.groupby("_cluster")["label"].nunique()
    single_clusters = label_nunique[label_nunique == 1].index
    single = work[work["_cluster"].isin(single_clusters)]
    single = (
        single.sort_values(["_cluster", "_trust_priority", "_text_len"], ascending=[True, True, False])
        .drop_duplicates("_cluster", keep="first")
    )

    conflicting_clusters = label_nunique[label_nunique > 1].index
    conflicting = work[work["_cluster"].isin(conflicting_clusters)]
    kept_conflicting = pd.DataFrame(columns=work.columns)
    dropped_conflicts = 0
    if len(conflicting):
        majority = conflicting.groupby("_cluster")["label"].agg(
            lambda s: int(s.value_counts().idxmax())
        )
        share = conflicting.groupby("_cluster")["label"].agg(
            lambda s: float(s.value_counts().max()) / float(len(s))
        )
        sizes = conflicting.groupby("_cluster").size()
        keep_clusters = share[(share >= 0.8) & (sizes >= 3)].index
        if len(keep_clusters):
            sub = conflicting[conflicting["_cluster"].isin(keep_clusters)]
            sub = sub[sub["label"] == sub["_cluster"].map(majority)]
            kept_conflicting = (
                sub.sort_values(["_cluster", "_trust_priority", "_text_len"], ascending=[True, True, False])
                .drop_duplicates("_cluster", keep="first")
            )
        dropped_conflicts = int(len(conflicting_clusters) - kept_conflicting["_cluster"].nunique())

    out = pd.concat([single, kept_conflicting], ignore_index=False).sort_index()
    is_synth = out["source"].eq("synthetic")
    out["group"] = np.where(
        is_synth,
        out["group"],
        "dedupe:" + out["_cluster"].astype(str),
    )
    stats = {
        "input_rows": int(len(work)),
        "output_rows": int(len(out)),
        "rows_removed": int(len(work) - len(out)),
        "clusters": int(work["_cluster"].nunique()),
        "conflicting_clusters_dropped": dropped_conflicts,
        "conflicting_rows": int(len(conflicting)),
        "conflicting_clusters_kept": int(kept_conflicting["_cluster"].nunique()) if len(kept_conflicting) else 0,
    }
    out = out.drop(columns=["_cluster", "_trust_priority", "_text_len"])
    return out, stats


def _majority(series: pd.Series):
    mode = series.mode()
    return mode.iloc[0] if len(mode) else series.iloc[0]


def _group_split(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    seed = int(config["project"]["seed"])
    train_ratio = float(config["data"]["train_ratio"])
    val_ratio = float(config["data"]["val_ratio"])
    test_ratio = float(config["data"]["test_ratio"])
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    group_stats = df.groupby("group", sort=False).agg(
        label=("label", _majority),
        language=("language", _majority),
        n=("text", "size"),
    ).reset_index()
    group_stats["strat"] = group_stats["label"].astype(str) + "|" + group_stats["language"].astype(str)
    strat_counts = group_stats["strat"].value_counts()
    rare = strat_counts[strat_counts < 3].index
    group_stats.loc[group_stats["strat"].isin(rare), "strat"] = "rare"
    group_stats = group_stats.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    try:
        train_groups, temp_groups = train_test_split(
            group_stats,
            test_size=(1.0 - train_ratio),
            stratify=group_stats["strat"],
            random_state=seed,
        )
    except ValueError:
        train_groups, temp_groups = train_test_split(
            group_stats, test_size=(1.0 - train_ratio), random_state=seed
        )
    try:
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_groups["strat"],
            random_state=seed,
        )
    except ValueError:
        val_groups, test_groups = train_test_split(
            temp_groups, test_size=0.5, random_state=seed
        )

    mapping = {}
    for name, frame in (("train", train_groups), ("val", val_groups), ("test", test_groups)):
        mapping.update({g: name for g in frame["group"]})

    out = df.copy()
    out["split"] = out["group"].map(mapping)
    missing = out["split"].isna().sum()
    if missing:
        out.loc[out["split"].isna(), "split"] = "train"

    group_overlap = out.groupby("group")["split"].nunique()
    out = out.sort_values(["split", "label", "language"], kind="stable").reset_index(drop=True)

    hash_sets = {
        name: set(out.loc[out["split"] == name, "text"].map(text_hash))
        for name in ("train", "val", "test")
    }
    stats = {
        "groups": int(group_stats.shape[0]),
        "groups_with_overlap": int((group_overlap > 1).sum()),
        "split_rows": {k: int(v) for k, v in out["split"].value_counts().items()},
        "exact_hash_overlap_train_val": len(hash_sets["train"] & hash_sets["val"]),
        "exact_hash_overlap_train_test": len(hash_sets["train"] & hash_sets["test"]),
        "exact_hash_overlap_val_test": len(hash_sets["val"] & hash_sets["test"]),
    }
    return out, stats


def _realistic_subset(df: pd.DataFrame, spam_prior: float, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    ham = df[df["label"] == 0]
    spam = df[df["label"] == 1]
    if len(ham) == 0 or len(spam) == 0:
        return df.sample(frac=1.0, random_state=seed)
    target_spam = int(round(len(ham) * spam_prior / max(1e-9, 1.0 - spam_prior)))
    n_spam = min(target_spam, len(spam))
    parts = [ham]
    if n_spam > 0:
        parts.append(spam.sample(n=n_spam, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _write_data_card(path: Path, stats: dict, config: dict) -> None:
    lines = [
        "# Dataset Card — Multilingual Spam & Scam Detection",
        "",
        f"Generated with seed **{config['project']['seed']}**.",
        "",
        "## Sources",
        "",
    ]
    for key, info in SOURCE_INFO.items():
        lines.append(f"- **{info['title']}** (`{key}`): {info['note']} License: {info['license']}.")
    lines.extend([
        "",
        "## Leakage controls",
        "",
        "- All sources were merged before splitting; the original ScamShield/UCI splits were not reused.",
        "- Exact duplicates and SimHash near-duplicates were removed globally before splitting.",
        "- Splitting is group-aware: synthetic template groups never cross train/val/test.",
        "- Stratification uses the `label|language` key with rare strata merged.",
        "",
        "## Split summary",
        "",
        "```text",
        json.dumps(stats.get("split", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Label distribution",
        "",
        "```text",
        json.dumps(stats.get("label_by_split", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Language distribution",
        "",
        "```text",
        json.dumps(stats.get("language_by_split", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Data quality notes",
        "",
        "- Some real-world corpora contain label noise. Obvious contradictory noisy-source rows were filtered with the heuristic audit; ambiguous clusters were dropped during deduplication.",
        "- The `Kaggle_Hindi_Merged` subset from ScamShield was excluded because its texts were low quality and heavily skewed.",
        "- Synthetic hard negatives deliberately include money requests, legitimate URLs, OTP notices, bank alerts and urgent family messages.",
        "- This dataset is for research and defensive spam filtering. Check each upstream license before commercial use.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset(config_path: str | Path | None = None, force: bool = False) -> dict:
    config = load_config(config_path)
    set_seed(int(config["project"]["seed"]))
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    processed_dir = ensure_dir(config["paths"]["processed_dir"])
    artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])
    train_path = processed_dir / "train.csv"
    if not force and train_path.exists() and (processed_dir / "dataset_stats.json").exists():
        with open(processed_dir / "dataset_stats.json", "r", encoding="utf-8") as fh:
            stats = json.load(fh)
        print(f"Dataset already exists at {processed_dir}. Use --force to rebuild.")
        return stats

    frames: list[pd.DataFrame] = []
    raw_stats: dict[str, dict] = {}
    for key, loader, cap_key in LOADERS:
        print(f"Loading {key}...", flush=True)
        try:
            frame = loader(config)
        except Exception as exc:
            print(f"  WARNING: failed to load {key}: {exc}", flush=True)
            raw_stats[key] = {"rows": 0, "error": str(exc)}
            continue
        cap = int(config["data"].get("source_caps", {}).get(cap_key, 0) or 0)
        before = len(frame)
        if cap > 0:
            frame = _cap_source(frame, cap, int(config["project"]["seed"]))
        raw_stats[key] = {
            "rows_loaded": int(before),
            "rows_after_cap": int(len(frame)),
            "spam": int(frame["label"].sum()) if len(frame) else 0,
            "ham": int((frame["label"] == 0).sum()) if len(frame) else 0,
        }
        print(f"  {before:,} loaded, {len(frame):,} kept after cap", flush=True)
        if len(frame):
            frames.append(frame)

    print("Loading synthetic augmentation...", flush=True)
    synth = load_synthetic(config)
    raw_stats["synthetic"] = {
        "rows_loaded": int(len(synth)),
        "rows_after_cap": int(len(synth)),
        "spam": int(synth["label"].sum()) if len(synth) else 0,
        "ham": int((synth["label"] == 0).sum()) if len(synth) else 0,
    }
    if len(synth):
        frames.append(synth)

    if not frames:
        raise RuntimeError("No datasets were loaded. Check data/raw and config.yaml.")

    df = pd.concat(frames, ignore_index=True)
    print(f"Combined raw rows: {len(df):,}", flush=True)

    min_chars = int(config["data"].get("min_text_chars", 3))
    max_chars = int(config["data"].get("max_text_chars", 1200))
    df["text"] = df["text"].map(clean_text)
    df["text"] = df["text"].str.slice(0, max_chars)
    before_basic = len(df)
    df = df[df["text"].str.len() >= min_chars]
    df = df[df["text"].str.contains(r"[^\W\d_]", regex=True, na=False)]
    df = df[df["label"].isin([0, 1])]
    basic_stats = {"input": int(before_basic), "output": int(len(df)), "dropped": int(before_basic - len(df))}

    print("Auditing noisy-source label quality...", flush=True)
    df, quality_stats = _quality_filter(df, config)

    print("Removing exact and near duplicates...", flush=True)
    df, dedupe_stats = _dedupe(df, config)
    print(f"  kept {len(df):,} rows after dedupe", flush=True)

    print("Building group-aware leakage-free splits...", flush=True)
    df, split_stats = _group_split(df, config)
    print(f"  split sizes: {split_stats['split_rows']}", flush=True)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    prior = float(config["data"].get("realistic_spam_prior", 0.20))
    seed = int(config["project"]["seed"])
    val_real = _realistic_subset(val, prior, seed)
    test_real = _realistic_subset(test, prior, seed)
    val_cal, val_tune = train_test_split(
        val_real, test_size=0.5, stratify=val_real["label"] if val_real["label"].nunique() > 1 else None,
        random_state=seed,
    )

    output_cols = ["text", "label", "language", "category", "source", "group", "split", "trust", "raw_id"]
    for name, frame in (("train", train), ("val", val), ("test", test),
                        ("val_cal", val_cal), ("val_tune", val_tune),
                        ("test_realistic", test_real)):
        frame = frame[output_cols].reset_index(drop=True)
        frame.to_csv(processed_dir / f"{name}.csv", index=False, encoding="utf-8")

    label_by_split = {
        name: {
            "rows": int(len(frame)),
            "spam": int(frame["label"].sum()),
            "ham": int((frame["label"] == 0).sum()),
            "spam_prior": round(float(frame["label"].mean()), 4) if len(frame) else 0.0,
        }
        for name, frame in (("train", train), ("val", val), ("test", test),
                            ("val_cal", val_cal), ("val_tune", val_tune),
                            ("test_realistic", test_real))
    }
    language_by_split = {
        name: {str(k): int(v) for k, v in frame["language"].value_counts().items()}
        for name, frame in (("train", train), ("val", val), ("test", test))
    }
    source_by_label = {
        str(source): {
            "rows": int(len(group)),
            "spam": int(group["label"].sum()),
            "ham": int((group["label"] == 0).sum()),
        }
        for source, group in df.groupby("source", sort=False)
    }
    category_counts = {
        str(cat): int(n) for cat, n in df["category"].value_counts().head(40).items()
    }

    stats = {
        "project": config["project"]["name"],
        "version": config["project"]["version"],
        "seed": seed,
        "raw_sources": raw_stats,
        "basic_filter": basic_stats,
        "quality_filter": quality_stats,
        "dedupe": dedupe_stats,
        "split": split_stats,
        "label_by_split": label_by_split,
        "language_by_split": language_by_split,
        "source_by_label": source_by_label,
        "top_categories": category_counts,
        "realistic_priors": {
            "val_cal": round(float(val_cal["label"].mean()), 4) if len(val_cal) else 0.0,
            "val_tune": round(float(val_tune["label"].mean()), 4) if len(val_tune) else 0.0,
            "test_realistic": round(float(test_real["label"].mean()), 4) if len(test_real) else 0.0,
        },
    }
    with open(processed_dir / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    _write_data_card(processed_dir / "data_card.md", stats, config)

    print("\nLabel by split:")
    print(pd.DataFrame(label_by_split).T.to_string())
    print("\nLanguage by split:")
    print(pd.DataFrame(language_by_split).fillna(0).astype(int).T.to_string())
    print("\nTop sources:")
    print(pd.DataFrame(source_by_label).T.sort_values("rows", ascending=False).head(15).to_string())
    print(f"\nSaved processed datasets to {processed_dir}")
    print(f"LEAKAGE: exact hash overlaps={split_stats['exact_hash_overlap_train_val']}/"
          f"{split_stats['exact_hash_overlap_train_test']}/"
          f"{split_stats['exact_hash_overlap_val_test']}, "
          f"group overlaps={split_stats['groups_with_overlap']}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the multilingual spam dataset.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    stats = build_dataset(args.config, force=args.force)
    print(json.dumps({
        "rows": stats.get("split", {}).get("split_rows"),
        "priors": stats.get("realistic_priors"),
        "dedupe": stats.get("dedupe"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
