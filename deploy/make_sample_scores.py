from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.model_builder import predict_lexical
from src.utils import load_config


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv").sample(40, random_state=7)
    texts = val["text"].astype(str).tolist()
    texts += list(config["ui"]["examples"])
    texts += [
        "Your OTP is 482913. Do not share it with anyone.",
        "Bro can you send me $5 for lunch? I'll pay you back tomorrow.",
        "আপনি ৫,০০,০০০ টাকা জিতেছেন! পুরস্কার পেতে আপনার বিকাশ নম্বর পাঠান।",
        "Aapka KYC pending hai, turant is link par click karke update karein: bit[.]ly/kyc-update",
        "URGENT: Your bKash account is suspended. Click http://bkash-verify.xyz to reactivate.",
        "I sent you $500 by mistake. Please send $300 back and keep the rest.",
        "Your parcel is held at customs. Pay the 2.99 delivery fee here: http://dhl-fee.top",
        "Hello, are we still meeting for coffee at 5?",
    ]
    lexical = joblib.load(ROOT / "artifacts" / "lexical_model.joblib")
    lr_probs, svm_scores, _ = predict_lexical(lexical, texts)
    payload = {
        "texts": texts,
        "lr": [float(x) for x in lr_probs],
        "svm": [float(x) for x in svm_scores],
    }
    path = ROOT / "deploy" / "sample_scores.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path} with {len(texts)} texts")


if __name__ == "__main__":
    main()
