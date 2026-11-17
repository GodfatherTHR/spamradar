from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from .features import FEATURE_NAMES, OTP_SAFE_LOCAL_RE, analyze_text
from .model_builder import (
    SemanticEncoder,
    apply_calibrator,
    heuristic_scores,
    predict_lexical,
)
from .utils import clean_text, load_config, resolve_path

SEMANTIC_PROTOTYPES = [
    ("p2p_bait", "Send me a small amount of money and I will give you a valuable item or double your money back."),
    ("p2p_bait", "Give me five dollars and I will give you an RTX graphics card. Only one left."),
    ("p2p_bait", "Bhai mujhe paise bhej do, main tumhe double wapas dunga."),
    ("crypto", "Send cryptocurrency to this wallet and receive double the amount back within minutes. Guaranteed profit."),
    ("bank_kyc", "Your bank account is suspended due to suspicious activity. Verify your identity through this link immediately."),
    ("bank_kyc", "Your KYC is pending and your account will be blocked. Update your details at this link."),
    ("lottery_prize", "You won a lottery or prize. Pay a processing fee to claim your reward now."),
    ("delivery_otp", "Your parcel is held at customs. Pay a small delivery fee using this link to release it."),
    ("delivery_otp", "Delivery failed. Share the OTP you received with our courier to reschedule the parcel."),
    ("job", "Work from home and earn a daily payout with no experience. Contact us on WhatsApp or Telegram."),
    ("romance", "I am a soldier overseas and I need money for a visa or hospital bill so we can finally meet."),
    ("marketplace", "I am selling an expensive item from overseas. Send a deposit first and I will ship it to you."),
    ("loan_investment", "Guaranteed high returns in a few days. Invest a small amount and double your money."),
    ("tech_support", "Your device is infected. Call this number and allow remote access so we can fix it."),
    ("promo", "Limited time offer. Click this link now to claim a free gift or huge discount."),
    ("otp_theft", "Tell me the verification code or OTP you just received so I can verify your account."),
    ("bank_alert", "Your bank account was debited by a card payment. The available balance is shown below."),
    ("otp", "Your OTP is a private verification code. Do not share it with anyone."),
    ("personal", "Can you send me a small amount for lunch? I will pay you back tomorrow."),
    ("delivery", "Your package is out for delivery and should arrive today. Track the parcel here."),
    ("personal", "A family member is in the hospital. Please call as soon as you see this message."),
]

GENERIC_CUE_NAMES = {
    "alpha ratio", "upper ratio", "len chars", "digit ratio", "punct ratio",
    "len tokens", "emoji count", "exclamation count", "question count",
    "short text flag", "long text flag", "script latin share",
    "script devanagari share", "script bengali share", "script arabic share",
    "script other share", "mixed script flag", "code mixed flag",
    "url early flag", "emoji flag", "url present flag",
}

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


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _clean_feature_name(name: str) -> str:
    value = name.split(":", 1)[-1]
    replacements = {
        "__url__": "URL pattern", "__email__": "email", "__phone__": "phone",
        "__amount__": "money amount", "__otp__": "OTP", "__num__": "number",
    }
    for key, human in replacements.items():
        value = value.replace(key, human)
    value = value.replace("_", " ").strip()
    if len(value) > 48:
        value = value[:48]
    return value


class SpamPredictor:
    def __init__(
        self,
        config_path: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
        load_semantic: bool = True,
        prefer_onnx: bool = True,
        warmup: bool = True,
    ):
        self.config = load_config(config_path)
        self.project_root = resolve_path(".")
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else resolve_path(
            self.config["paths"]["artifacts_dir"]
        )
        self.version = str(self.config["project"]["version"])
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.started_at = time.time()

        self.lexical = self._load_required("lexical_model.joblib")
        self.calibrator = self._load_required("calibrator.joblib")
        self.thresholds = _load_json_if_exists(self.artifacts_dir / "threshold_config.json") or {
            "spam_threshold": 0.62,
            "suspicious_threshold": 0.30,
        }
        self.metadata = _load_json_if_exists(self.artifacts_dir / "model_metadata.json") or {}
        self._semantic_expected = "semantic_logit" in self.calibrator.get("feature_names", [])
        self.semantic = None
        self.semantic_error = None
        self._prototype_embeddings = None
        self._prototype_texts = [p[1] for p in SEMANTIC_PROTOTYPES]
        self._prototype_families = [p[0] for p in SEMANTIC_PROTOTYPES]
        self._prototype_is_scam = [f not in {"bank_alert", "otp", "personal", "delivery"} for f in self._prototype_families]

        if load_semantic:
            self._load_semantic(prefer_onnx=prefer_onnx)
        if self.semantic is not None:
            self._prepare_prototypes()
        if warmup:
            self.warmup()

    def _load_required(self, name: str):
        path = self.artifacts_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact {path}. Run `python -m src.train` first."
            )
        return joblib.load(path)

    def _load_semantic(self, prefer_onnx: bool = True) -> None:
        head_path = self.artifacts_dir / "semantic_head.joblib"
        meta_path = self.artifacts_dir / "semantic_metadata.json"
        if not head_path.exists() or not meta_path.exists():
            return
        try:
            payload = joblib.load(head_path)
            meta = _load_json_if_exists(meta_path) or {}
            model_name = meta.get("model_name") or payload.get("model_name")
            max_seq_length = int(meta.get("max_seq_length", 128))
            batch_size = int(meta.get("batch_size", 64))
            onnx_path = self.artifacts_dir / "semantic_encoder_int8.onnx"
            if not onnx_path.exists():
                onnx_path = self.artifacts_dir / "semantic_encoder.onnx"
            use_onnx = prefer_onnx and onnx_path.exists()
            tokenizer_dir = self.artifacts_dir / "tokenizer"
            encoder = SemanticEncoder(
                model_name=model_name,
                onnx_path=onnx_path if use_onnx else None,
                tokenizer_dir=tokenizer_dir if tokenizer_dir.exists() else None,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
            )
            encoder.encode(["warmup"])
            self.semantic = {
                "encoder": encoder,
                "head": payload["head"],
                "scaler": payload["scaler"],
                "model_name": model_name,
                "max_seq_length": max_seq_length,
                "batch_size": batch_size,
                "backend": "onnx-int8" if use_onnx and "int8" in onnx_path.name else (
                    "onnx" if use_onnx else "pytorch"
                ),
                "embedding_dim": int(payload.get("embedding_dim", 384)),
            }
        except Exception as exc:
            self.semantic_error = f"{type(exc).__name__}: {exc}"
            self.semantic = None

    def _prepare_prototypes(self) -> None:
        try:
            embeddings = self.semantic["encoder"].encode(self._prototype_texts)
            self._prototype_embeddings = embeddings.astype(np.float32)
        except Exception as exc:
            self.semantic_error = f"prototype encoder failed: {type(exc).__name__}: {exc}"
            self._prototype_embeddings = None

    def _semantic_similarity(self, embedding: np.ndarray | None) -> dict | None:
        if embedding is None or self._prototype_embeddings is None:
            return None
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm <= 0:
            return None
        vector = vector / norm
        similarities = self._prototype_embeddings @ vector
        order = np.argsort(similarities)[::-1]
        for idx in order:
            if self._prototype_is_scam[int(idx)]:
                return {
                    "similarity": round(float(similarities[int(idx)]), 4),
                    "prototype": self._prototype_texts[int(idx)],
                    "family": self._prototype_families[int(idx)],
                }
        return None

    def _thresholds(self) -> tuple[float, float]:
        return (
            float(self.thresholds.get("suspicious_threshold", 0.30)),
            float(self.thresholds.get("spam_threshold", 0.62)),
        )

    def risk_label(self, probability: float) -> tuple[str, str]:
        low, high = self._thresholds()
        if probability >= high:
            return "SPAM", "high"
        if probability >= low:
            return "SUSPICIOUS", "review"
        return "NORMAL", "low"

    def _truncate(self, text: object) -> str:
        value = "" if text is None else str(text)
        limit = int(self.config.get("api", {}).get("max_text_length", 4000))
        return value[:limit]

    def score_texts(
        self,
        texts: Sequence[str],
        embeddings: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> dict:
        items = [self._truncate(t) for t in texts]
        if not items:
            return {
                "final": np.empty(0), "lexical_lr": np.empty(0),
                "lexical_svm": np.empty(0), "semantic": None,
                "heuristic": np.empty(0), "embeddings": None,
            }
        lr_probs, svm_scores, _ = predict_lexical(self.lexical, items)
        sem_probs = None
        sem_embeddings = None
        if self.semantic is not None:
            if embeddings is not None:
                sem_embeddings = np.asarray(embeddings, dtype=np.float32)
            else:
                sem_embeddings = self.semantic["encoder"].encode(
                    items, batch_size=batch_size, show_progress=False
                )
            sem_probs = self.semantic["head"].predict_proba(
                self.semantic["scaler"].transform(sem_embeddings)
            )[:, 1]
        h = heuristic_scores(items)
        if self._semantic_expected:
            sem_for_cal = sem_probs if sem_probs is not None else np.full(len(items), 0.5)
        else:
            sem_for_cal = None
        final = apply_calibrator(
            self.calibrator, lr_probs, svm_scores, sem_for_cal, h
        )
        return {
            "final": final,
            "lexical_lr": lr_probs,
            "lexical_svm": svm_scores,
            "semantic": sem_probs,
            "heuristic": h,
            "embeddings": sem_embeddings,
        }

    def _rule_decision(
        self,
        analysis,
        probability: float,
        label: str,
        risk: str,
    ) -> tuple[str, str, str, list[str]]:
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
        return label, risk, "calibrated_model", []

    def _build_result(
        self,
        text: str,
        probability: float,
        lex_lr: float,
        lex_svm: float,
        sem_prob: float | None,
        heuristic: float,
        analysis,
        cues: dict,
        semantic_sim: dict | None,
        latency_ms: float,
    ) -> dict:
        label, risk = self.risk_label(float(probability))
        label, risk, decision_source, rule_reasons = self._rule_decision(
            analysis, float(probability), label, risk
        )
        signals = list(analysis.signals)
        if semantic_sim and semantic_sim["similarity"] >= 0.60:
            signals.append({
                "name": "semantic_scam_match",
                "title": "Semantically close to a known scam script",
                "weight": round(min(0.35, 0.15 + semantic_sim["similarity"] * 0.2), 4),
                "severity": "medium",
                "evidence": [semantic_sim["prototype"]],
            })
        if rule_reasons:
            signals.append({
                "name": "rule_escalation",
                "title": "High-confidence rule: " + rule_reasons[0],
                "weight": 0.65 if decision_source == "rule_escalation" else 0.28,
                "severity": "high" if decision_source == "rule_escalation" else "medium",
                "evidence": rule_reasons[:3],
            })
        signals.sort(key=lambda s: s.get("weight", 0), reverse=True)
        signal_titles = [
            s["title"] for s in signals
            if s.get("weight", 0) >= 0.12 and s.get("name") != "rule_escalation"
        ][:3]
        if decision_source == "rule_escalation":
            explanation = "High-confidence scam rule: " + "; ".join(rule_reasons[:3]) + "."
            if signal_titles:
                explanation += " Model signals: " + ", ".join(signal_titles[:2]) + "."
        elif decision_source == "rule_review":
            explanation = "Review recommended: " + "; ".join(rule_reasons[:2]) + "."
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
        risk_cues = [
            {
                "feature": _clean_feature_name(c["feature"]),
                "raw_feature": c["feature"],
                "contribution": c["contribution"],
            }
            for c in cues.get("risk_cues", [])
            if c.get("contribution", 0) > 0
        ]
        specific_cues = [c for c in risk_cues if c["feature"] not in GENERIC_CUE_NAMES]
        generic_cues = [c for c in risk_cues if c["feature"] in GENERIC_CUE_NAMES]
        risk_cues = (specific_cues + generic_cues)[:6]
        low, high = self._thresholds()
        return {
            "label": label,
            "risk_tier": risk,
            "spam_probability": round(float(probability), 6),
            "thresholds": {
                "suspicious": round(low, 6),
                "spam": round(high, 6),
            },
            "language": analysis.language,
            "language_confidence": round(float(analysis.language_confidence), 4),
            "detected_signals": signals,
            "model_cues": risk_cues,
            "semantic_match": semantic_sim,
            "scores": {
                "lexical_logistic": round(float(lex_lr), 6),
                "lexical_svm": round(float(lex_svm), 6),
                "semantic": round(float(sem_prob), 6) if sem_prob is not None else None,
                "heuristic": round(float(heuristic), 6),
            },
            "latency_ms": round(float(latency_ms), 3),
            "text_length": len(text),
            "explanation": explanation,
            "decision": {
                "source": decision_source,
                "rule_reasons": rule_reasons,
                "calibrated_probability": round(float(probability), 6),
                "effective_label": label,
            },
            "model_version": self.version,
        }

    def predict(self, text: object) -> dict:
        started = time.perf_counter()
        value = self._truncate(text)
        analysis = analyze_text(value)
        scores = self.score_texts([value], batch_size=1)
        embedding = scores["embeddings"][0] if scores["embeddings"] is not None else None
        semantic_sim = self._semantic_similarity(embedding)
        coef = self.lexical["lr"].coef_[0]
        cues = self.lexical["feature_extractor"].explain(value, coef, top_k=8)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._build_result(
            text=value,
            probability=float(scores["final"][0]),
            lex_lr=float(scores["lexical_lr"][0]),
            lex_svm=float(scores["lexical_svm"][0]),
            sem_prob=float(scores["semantic"][0]) if scores["semantic"] is not None else None,
            heuristic=float(scores["heuristic"][0]),
            analysis=analysis,
            cues=cues,
            semantic_sim=semantic_sim,
            latency_ms=latency_ms,
        )

    def predict_batch(self, texts: Sequence[str], batch_size: int | None = None) -> list[dict]:
        items = [self._truncate(t) for t in texts]
        started = time.perf_counter()
        scores = self.score_texts(items, batch_size=batch_size)
        coef = self.lexical["lr"].coef_[0]
        results = []
        for i, text in enumerate(items):
            analysis = analyze_text(text)
            embedding = scores["embeddings"][i] if scores["embeddings"] is not None else None
            semantic_sim = self._semantic_similarity(embedding)
            cues = self.lexical["feature_extractor"].explain(text, coef, top_k=6)
            results.append(self._build_result(
                text=text,
                probability=float(scores["final"][i]),
                lex_lr=float(scores["lexical_lr"][i]),
                lex_svm=float(scores["lexical_svm"][i]),
                sem_prob=float(scores["semantic"][i]) if scores["semantic"] is not None else None,
                heuristic=float(scores["heuristic"][i]),
                analysis=analysis,
                cues=cues,
                semantic_sim=semantic_sim,
                latency_ms=0.0,
            ))
        total_ms = (time.perf_counter() - started) * 1000.0
        per_item = total_ms / max(1, len(items))
        for result in results:
            result["latency_ms"] = round(float(per_item), 3)
        return results

    def health(self) -> dict:
        return {
            "status": "ok" if self.lexical is not None else "degraded",
            "model_version": self.version,
            "model_type": self.metadata.get("model_type", "hybrid"),
            "backbone": self.semantic["model_name"] if self.semantic else None,
            "semantic_backend": self.semantic["backend"] if self.semantic else "disabled",
            "semantic_error": self.semantic_error,
            "thresholds": self.thresholds,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "device": "cpu",
            "artifacts_dir": str(self.artifacts_dir),
        }

    def stats(self) -> dict:
        processed_dir = resolve_path(self.config["paths"]["processed_dir"])
        evaluation = _load_json_if_exists(self.artifacts_dir / "evaluation_metrics.json")
        dataset = _load_json_if_exists(processed_dir / "dataset_stats.json")
        benchmark = _load_json_if_exists(self.artifacts_dir / "onnx_benchmark.json")
        return {
            "model": self.metadata,
            "thresholds": self.thresholds,
            "semantic": {
                "model_name": self.semantic["model_name"],
                "backend": self.semantic["backend"],
                "embedding_dim": self.semantic["embedding_dim"],
            } if self.semantic else None,
            "evaluation": evaluation,
            "dataset": dataset,
            "onnx_benchmark": benchmark,
        }

    def examples(self) -> list[str]:
        return list(self.config.get("ui", {}).get("examples", []))

    def warmup(self) -> None:
        try:
            self.score_texts(["hello, how are you?"], batch_size=1)
            self.predict("hello, how are you?")
        except Exception:
            pass
