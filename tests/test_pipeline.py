from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_NAMES,
    analyze_text,
    detect_language,
    prepare_for_model,
)
from src.synthetic import generate_synthetic, validate_templates
from src.utils import (
    canonical_for_dedupe,
    clean_text,
    extract_amounts,
    extract_otps,
    extract_urls,
    load_config,
    refang,
    text_hash,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()
ARTIFACTS = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed"


def test_clean_text_removes_control_and_zero_width():
    raw = "Hello\u200b   world\n\nthis is\u200c fine"
    cleaned = clean_text(raw)
    assert "\u200b" not in cleaned
    assert "\u200c" not in cleaned
    assert "Hello world this is fine" == cleaned


def test_refang_defangs_urls():
    text = "Pay at bit[.]ly/kyc and visit hxxps://paypa1-secure(.)top/login"
    refanged = refang(text)
    assert "bit.ly/kyc" in refanged
    assert "https://paypa1-secure.top/login" in refanged


def test_extract_urls_shortener_and_ip():
    urls = extract_urls("Track at https://bit.ly/abc123 or http://192.168.1.44/login")
    domains = {url.domain for url in urls}
    assert "bit.ly" in domains
    assert any(url.is_ip for url in urls)
    assert any(url.is_shortener for url in urls)


def test_extract_urls_spaced_letters():
    urls = extract_urls("Your gift is at w.a.l.m.a.r.t/claim")
    assert any(url.is_obfuscated for url in urls)


def test_amounts_and_otp_extraction():
    assert extract_amounts("Send $5 today and get 0.5 BTC")[0]["value"] == 5.0
    assert extract_amounts("আপনি ৫,০০,০০০ টাকা জিতেছেন")[0]["value"] == 500000.0
    assert "482913" in extract_otps("Your OTP is 482913. Do not share it.")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Your OTP is 482913. Do not share it.", "English"),
        ("আপনি ৫,০০,০০০ টাকা জিতেছেন।", "Bengali"),
        ("Aapka KYC pending hai, turant link par click karein.", "Hinglish"),
        ("Bhai 500 taka bhej, ami 5000 taka debo.", "Banglish"),
        ("Mi cuenta ha sido bloqueada, verifique aquí.", "Spanish"),
        ("Votre compte est suspendu, cliquez ici.", "French"),
        ("حسابك موقوف، انقر هنا.", "Arabic"),
        ("आपका OTP 482913 है। इसे साझा न करें।", "Hindi"),
    ],
)
def test_language_detection(text, expected):
    language, confidence, shares = detect_language(text)
    assert language == expected
    assert 0.0 <= confidence <= 1.0


def test_prepare_for_model_placeholders():
    prepared = prepare_for_model("Your OTP is 482913 at http://bit.ly/x, email a@b.com")
    assert "__otp__" in prepared or "__num__" in prepared
    assert "__url__" in prepared
    assert "__email__" in prepared


def test_p2p_scam_signal():
    analysis = analyze_text("Give me $5 bucks I will give you an RTX 3090")
    names = {s["name"] for s in analysis.signals}
    assert "unrealistic_exchange" in names


def test_normal_lunch_is_not_p2p_bait():
    analysis = analyze_text(
        "Bro can you send me $5 for lunch? I will pay you back tomorrow."
    )
    names = {s["name"] for s in analysis.signals}
    assert "unrealistic_exchange" not in names


def test_legitimate_otp_is_not_otp_theft():
    analysis = analyze_text("Your OTP is 482913. Do not share it with anyone.")
    names = {s["name"] for s in analysis.signals}
    assert "otp_theft" not in names


def test_otp_request_is_otp_theft():
    analysis = analyze_text(
        "Send me the OTP you received to verify your account."
    )
    names = {s["name"] for s in analysis.signals}
    assert "otp_theft" in names


def test_bank_phishing_signal():
    analysis = analyze_text(
        "URGENT: Your bKash account is suspended. Click http://bkash-verify.xyz to reactivate."
    )
    names = {s["name"] for s in analysis.signals}
    assert "account_threat" in names
    assert "suspicious_url" in names


def test_feature_vector_is_finite_and_complete():
    analysis = analyze_text("Give me $5 and I will send a RTX 3090")
    vector = analysis.feature_vector
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vector).all()


def test_feature_names_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_synthetic_templates_validate():
    assert validate_templates() == []


def test_synthetic_generation_has_multilingual_labels():
    rows = generate_synthetic(CONFIG, seed=7)
    assert len(rows) > 1000
    languages = {row.language for row in rows}
    assert {"English", "Bengali", "Banglish", "Hindi", "Hinglish",
            "Spanish", "French", "Arabic", "mixed"}.issubset(languages)
    assert {row.label for row in rows} == {0, 1}


def test_adversarial_suite_schema():
    path = ROOT / "tests" / "adversarial_suite.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) >= 50
    labels = set()
    for row in rows:
        assert {"id", "text", "label", "language", "category", "why"}.issubset(row)
        assert row["label"] in {"spam", "ham"}
        assert len(row["text"]) > 5
        labels.add(row["label"])
    assert labels == {"spam", "ham"}
    assert sum(1 for row in rows if row["label"] == "spam") >= 25
    assert sum(1 for row in rows if row["label"] == "ham") >= 25


@pytest.mark.skipif(not (PROCESSED / "train.csv").exists(), reason="processed dataset missing")
def test_split_has_no_exact_hash_leakage():
    train = pd.read_csv(PROCESSED / "train.csv", usecols=["text", "group", "split"])
    val = pd.read_csv(PROCESSED / "val.csv", usecols=["text", "group", "split"])
    test = pd.read_csv(PROCESSED / "test.csv", usecols=["text", "group", "split"])
    hashes = {
        name: set(frame["text"].astype(str).map(text_hash))
        for name, frame in (("train", train), ("val", val), ("test", test))
    }
    assert hashes["train"].isdisjoint(hashes["val"])
    assert hashes["train"].isdisjoint(hashes["test"])
    assert hashes["val"].isdisjoint(hashes["test"])
    assert set(train["group"]).isdisjoint(set(val["group"]))
    assert set(train["group"]).isdisjoint(set(test["group"]))


@pytest.mark.skipif(not (PROCESSED / "train.csv").exists(), reason="processed dataset missing")
def test_adversarial_suite_has_no_exact_training_overlap():
    train = pd.read_csv(PROCESSED / "train.csv", usecols=["text"])
    train_hashes = set(train["text"].astype(str).map(text_hash))
    suite = json.loads((ROOT / "tests" / "adversarial_suite.json").read_text(encoding="utf-8"))
    overlaps = [row["id"] for row in suite if text_hash(row["text"]) in train_hashes]
    assert overlaps == [], f"adversarial cases leak into training: {overlaps[:10]}"


@pytest.mark.skipif(not (ARTIFACTS / "lexical_model.joblib").exists(), reason="model artifacts missing")
def test_predictor_artifact_smoke():
    from src.inference import SpamPredictor

    predictor = SpamPredictor(load_semantic=False, warmup=False)
    result = predictor.predict("Give me $5 bucks I will give you an RTX 3090")
    assert set(result) >= {
        "label", "risk_tier", "spam_probability", "language",
        "detected_signals", "model_cues", "scores", "latency_ms",
        "explanation", "model_version",
    }
    assert result["label"] in {"SPAM", "SUSPICIOUS", "NORMAL"}
    assert 0.0 <= result["spam_probability"] <= 1.0
    assert predictor.health()["status"] == "ok"
    assert len(predictor.examples()) > 0


@pytest.mark.skipif(not (ARTIFACTS / "lexical_model.joblib").exists(), reason="model artifacts missing")
def test_predictor_ranks_scam_above_normal_lunch():
    from src.inference import SpamPredictor

    predictor = SpamPredictor(load_semantic=False, warmup=False)
    scam = predictor.predict("Give me $5 bucks I will give you an RTX 3090")
    normal = predictor.predict(
        "Bro can you send me $5 for lunch? I will pay you back tomorrow."
    )
    assert scam["spam_probability"] > normal["spam_probability"]
    assert normal["label"] != "SPAM"
    assert scam["label"] != "NORMAL"


@pytest.mark.skipif(not (ARTIFACTS / "lexical_model.joblib").exists(), reason="model artifacts missing")
def test_predictor_legitimate_otp_not_blocked():
    from src.inference import SpamPredictor

    predictor = SpamPredictor(load_semantic=False, warmup=False)
    result = predictor.predict("Your OTP is 482913. Do not share it with anyone.")
    assert result["label"] != "SPAM"


@pytest.mark.skipif(not (ARTIFACTS / "lexical_model.joblib").exists(), reason="model artifacts missing")
def test_predictor_batch_and_empty_text():
    from src.inference import SpamPredictor

    predictor = SpamPredictor(load_semantic=False, warmup=False)
    results = predictor.predict_batch([
        "Give me $5 bucks I will give you an RTX 3090",
        "Your OTP is 482913. Do not share it with anyone.",
        "সুপ্রভাত, আজ দেখা হবে?",
    ])
    assert len(results) == 3
    assert all(0.0 <= r["spam_probability"] <= 1.0 for r in results)
    empty = predictor.predict("")
    assert empty["label"] in {"SPAM", "SUSPICIOUS", "NORMAL"}
