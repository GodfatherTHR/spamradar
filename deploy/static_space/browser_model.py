from __future__ import annotations

import gzip
import json
import math
import re
import time
from pathlib import Path

from src.features import (
    FEATURE_NAMES,
    OTP_SAFE_LOCAL_RE,
    analyze_text,
    prepare_for_model,
)

BASE = Path(__file__).resolve().parent
WHITE_SPACES = re.compile(r"\s\s+")
WORD_RE = re.compile(r"(?u)\b\w+\b")

SEED_PHRASE_RE = re.compile(
    r"(?is)(?:(?:seed phrase|recovery phrase|12-word|24-word).{0,90}"
    r"(?:reply|send|share|give|tell)|"
    r"(?:reply|send|share|give|tell).{0,90}"
    r"(?:seed phrase|recovery phrase|12-word|24-word))"
)
MISTAKEN_TRANSFER_RE = re.compile(
    r"(?is)(?:accidentally|by mistake).{0,60}(?:sent|transferred)"
    r".{0,60}(?:send|return).{0,50}(?:back|me).{0,60}(?:keep|rest)"
)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _char_wb_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    text = WHITE_SPACES.sub(" ", text)
    out: list[str] = []
    for word in text.split():
        word = " " + word + " "
        upper = min(max_n, len(word))
        for n in range(min_n, upper + 1):
            for i in range(len(word) - n + 1):
                out.append(word[i : i + n])
    return out


def _word_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    tokens = WORD_RE.findall(text)
    out: list[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


class BrowserModel:
    def __init__(self) -> None:
        raw = (BASE / "browser_model.json.gz").read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        model = json.loads(raw.decode("utf-8"))
        self.model = model
        self.char_index = {ngram: i for i, ngram in enumerate(model["char_ngrams"])}
        self.word_index = {ngram: i for i, ngram in enumerate(model["word_ngrams"])}
        self.thresholds = _read_json(BASE / "browser_thresholds.json")
        self.spam = float(self.thresholds["spam_threshold"])
        self.suspicious = float(self.thresholds["suspicious_threshold"])
        self.signal_lr = model["signal_lr"]
        self.signal_svm = model["signal_svm"]
        self.signal_names = model["signal_names"]

    def _tfidf(self, text: str, index: dict, idf: list, min_n: int, max_n: int, char_wb: bool) -> dict[int, float]:
        ngrams = _char_wb_ngrams(text, min_n, max_n) if char_wb else _word_ngrams(text, min_n, max_n)
        counts: dict[int, int] = {}
        for ngram in ngrams:
            idx = index.get(ngram)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        values: dict[int, float] = {}
        for idx, count in counts.items():
            value = 1.0 + math.log(count) if self.model["sublinear_tf"] else float(count)
            value *= float(idf[idx])
            if value:
                values[idx] = value
        norm = math.sqrt(sum(value * value for value in values.values()))
        if norm > 0:
            values = {idx: value / norm for idx, value in values.items()}
        return values

    def _score(self, prepared: str, analysis) -> tuple[float, float, list[dict]]:
        model = self.model
        char_values = self._tfidf(
            prepared,
            self.char_index,
            model["char_idf"],
            int(model["char_ngram_range"][0]),
            int(model["char_ngram_range"][1]),
            True,
        )
        word_values = self._tfidf(
            prepared,
            self.word_index,
            model["word_idf"],
            int(model["word_ngram_range"][0]),
            int(model["word_ngram_range"][1]),
            False,
        )
        lr_decision = float(model["lr_intercept"])
        svm_decision = float(model["svm_intercept"])
        positive: list[tuple[float, str]] = []

        char_lr = model["char_lr"]
        char_svm = model["char_svm"]
        for idx, value in char_values.items():
            lr_part = value * float(char_lr[idx])
            svm_part = value * float(char_svm[idx])
            lr_decision += lr_part
            svm_decision += svm_part
            if lr_part > 0:
                positive.append((lr_part, "char:" + model["char_ngrams"][idx].strip()))

        word_lr = model["word_lr"]
        word_svm = model["word_svm"]
        for idx, value in word_values.items():
            lr_part = value * float(word_lr[idx])
            svm_part = value * float(word_svm[idx])
            lr_decision += lr_part
            svm_decision += svm_part
            if lr_part > 0:
                positive.append((lr_part, "word:" + model["word_ngrams"][idx]))

        signal_values = analysis.feature_vector
        for idx, value in enumerate(signal_values):
            value = float(value)
            if not value:
                continue
            lr_part = value * float(self.signal_lr[idx])
            lr_decision += lr_part
            svm_decision += value * float(self.signal_svm[idx])
            if lr_part > 0:
                positive.append((lr_part, "signal:" + self.signal_names[idx]))

        positive.sort(key=lambda item: item[0], reverse=True)
        cues = [
            {
                "feature": name,
                "contribution": round(float(value), 5),
            }
            for value, name in positive[:8]
        ]
        return lr_decision, svm_decision, cues

    def _risk_label(self, probability: float) -> tuple[str, str]:
        if probability >= self.spam:
            return "SPAM", "high"
        if probability >= self.suspicious:
            return "SUSPICIOUS", "review"
        return "NORMAL", "low"

    def _rule_decision(self, analysis, probability: float, label: str, risk: str):
        features = analysis.features
        text = analysis.text
        lowered = text.casefold()
        reasons: list[str] = []
        if features.get("unrealistic_exchange_flag"):
            reasons.append("unrealistic exchange or money-doubling pattern")
        if features.get("otp_theft_flag") and not OTP_SAFE_LOCAL_RE.search(text):
            reasons.append("remote OTP or verification-code request")
        if features.get("delivery_fee_flag") and features.get("suspicious_url_flag"):
            reasons.append("delivery/customs fee with a suspicious link")
        if features.get("account_threat_flag") and features.get("suspicious_url_flag"):
            reasons.append("account threat with a suspicious link")
        if features.get("kyc_flag") and features.get("suspicious_url_flag"):
            reasons.append("KYC update with a suspicious link")
        if features.get("crypto_scam_flag") and (
            features.get("suspicious_url_flag")
            or "seed phrase" in lowered
            or "recovery phrase" in lowered
        ):
            reasons.append("crypto doubling or wallet-recovery bait")
        if SEED_PHRASE_RE.search(text):
            reasons.append("seed-phrase request")
        if MISTAKEN_TRANSFER_RE.search(text):
            reasons.append("fake mistaken-transfer request")
        if (
            (features.get("prize_flag") or features.get("reward_count", 0) >= 2)
            and features.get("suspicious_url_flag")
            and (features.get("click_count") or "claim" in lowered)
        ):
            reasons.append("prize/reward claim with a suspicious link")
        if (
            features.get("marketplace_count")
            and (features.get("amount_count") or "deposit" in lowered)
            and any(k in lowered for k in (
                "zelle only", "cash app only", "deposit", "pay first",
                "advance", "ship it", "shipping", "will ship",
            ))
            and not any(k in lowered for k in (
                "pickup only", "no shipping", "local pickup", "meet in person",
            ))
        ):
            reasons.append("marketplace deposit/shipping trap")
        if (
            features.get("job_count")
            and (features.get("amount_count") or "fee" in lowered)
            and any(k in lowered for k in (
                "visa", "processing", "registration", "deposit", "training",
                "onboarding",
            ))
        ):
            reasons.append("job offer with an upfront fee")
        if (
            features.get("suspicious_url_flag")
            and features.get("authority_count")
            and any(k in lowered for k in (
                "confirm your payment", "verify your identity", "account limited",
                "order is on hold", "avoid cancellation", "fraud alert",
            ))
        ):
            reasons.append("brand phishing with a suspicious link")
        if (
            features.get("suspicious_url_flag")
            and ("refund" in lowered or "claim" in lowered)
            and (features.get("urgency_count") or features.get("threat_count") or "expires" in lowered)
        ):
            reasons.append("refund/claim with a suspicious link and urgency")
        if (
            features.get("phone_count")
            and any(k in lowered for k in (
                "virus", "remote access", "microsoft security", "apple id",
                "read the verification code",
            ))
        ):
            reasons.append("tech-support remote-access scam")
        if reasons:
            return "SPAM", "high", "rule_escalation", reasons

        review_reasons: list[str] = []
        romance_terms = (
            "my love", "deep connection", "feel a connection", "connection with you",
            "my dear", "soldier", "army", "widow", "peacekeeping", "valobasha",
            "bhalobasha", "pyaar", "mohabbat", "amour", "amor", "حب", "زواج",
            "প্রিয়", "সেনাবাহিনী", "ভালোবাসা",
        )
        if (
            features.get("romance_scam_flag")
            and (features.get("amount_count") or "money" in lowered)
            and any(k in lowered for k in romance_terms)
            and any(k in lowered for k in (
                "visa", "hospital", "ticket", "customs", "repay", "meet you", "দেখা",
            ))
        ):
            review_reasons.append("romance or emergency money request")
        if review_reasons and label == "NORMAL":
            return "SUSPICIOUS", "review", "rule_review", review_reasons
        return label, risk, "lexical_model", []

    def _build_result(self, text: str, probability: float, lr_prob: float, svm_score: float,
                      heuristic: float, analysis, cues: list[dict], latency_ms: float) -> dict:
        label, risk = self._risk_label(float(probability))
        label, risk, source, rule_reasons = self._rule_decision(
            analysis, float(probability), label, risk
        )
        signals = list(analysis.signals)
        if rule_reasons:
            signals.append({
                "name": "rule_escalation",
                "title": "High-confidence rule: " + rule_reasons[0],
                "weight": 0.65 if source == "rule_escalation" else 0.28,
                "severity": "high" if source == "rule_escalation" else "medium",
                "evidence": rule_reasons[:3],
            })
        signals.sort(key=lambda signal: signal.get("weight", 0), reverse=True)
        signal_titles = [
            signal["title"] for signal in signals
            if signal.get("weight", 0) >= 0.12 and signal.get("name") != "rule_escalation"
        ][:3]
        if source == "rule_escalation":
            explanation = "High-confidence scam rule: " + "; ".join(rule_reasons[:3]) + "."
            if signal_titles:
                explanation += " Model signals: " + ", ".join(signal_titles[:2]) + "."
        elif label == "SPAM":
            explanation = "High-risk message"
            if signal_titles:
                explanation += ": " + ", ".join(signal_titles)
            explanation += "."
        elif label == "SUSPICIOUS":
            explanation = "Review recommended"
            if signal_titles:
                explanation += ": " + ", ".join(signal_titles)
            explanation += "."
        else:
            explanation = "No strong scam signals were detected. The message resembles normal conversation or a legitimate notification."
            if signal_titles:
                explanation += " Low-level cues: " + ", ".join(signal_titles) + "."
        return {
            "label": label,
            "risk_tier": risk,
            "spam_probability": round(float(probability), 6),
            "thresholds": {
                "suspicious": round(self.suspicious, 6),
                "spam": round(self.spam, 6),
            },
            "language": analysis.language,
            "language_confidence": round(float(analysis.language_confidence), 4),
            "detected_signals": signals,
            "model_cues": cues,
            "semantic_match": None,
            "scores": {
                "lexical_logistic": round(float(lr_prob), 6),
                "lexical_svm": round(float(svm_score), 6),
                "semantic": None,
                "heuristic": round(float(heuristic), 6),
            },
            "latency_ms": round(float(latency_ms), 3),
            "text_length": len(text),
            "explanation": explanation,
            "decision": {
                "source": source,
                "rule_reasons": rule_reasons,
                "calibrated_probability": round(float(probability), 6),
                "effective_label": label,
            },
            "model_version": "1.0.0-browser-lexical",
        }

    def predict(self, text: object) -> dict:
        started = time.perf_counter()
        value = str(text or "")
        analysis = analyze_text(value)
        prepared = prepare_for_model(value)
        lr_decision, svm_decision, cues = self._score(prepared, analysis)
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, lr_decision))))
        heuristic = float(analysis.features.get("heuristic_score", 0.0))
        latency = (time.perf_counter() - started) * 1000.0
        return self._build_result(
            text=value,
            probability=probability,
            lr_prob=probability,
            svm_score=svm_decision,
            heuristic=heuristic,
            analysis=analysis,
            cues=cues,
            latency_ms=latency,
        )

    def predict_payload_json(self, payload_json: str) -> str:
        payload = json.loads(payload_json)
        return json.dumps(self.predict(payload.get("text", "")), ensure_ascii=False)

    def batch_payload_json(self, payload_json: str) -> str:
        payload = json.loads(payload_json)
        messages = [str(message) for message in payload.get("messages", [])]
        predictions = [self.predict(message) for message in messages]
        summary = {
            "spam": sum(1 for prediction in predictions if prediction["label"] == "SPAM"),
            "suspicious": sum(1 for prediction in predictions if prediction["label"] == "SUSPICIOUS"),
            "normal": sum(1 for prediction in predictions if prediction["label"] == "NORMAL"),
            "mean_latency_ms": round(
                sum(float(prediction["latency_ms"]) for prediction in predictions) / max(1, len(predictions)), 3
            ),
        }
        return json.dumps(
            {
                "count": len(predictions),
                "predictions": predictions,
                "summary": summary,
            },
            ensure_ascii=False,
        )
