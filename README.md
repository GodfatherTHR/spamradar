# SpamRadar

A multilingual spam and social engineering detector for short messages. It covers English, Bengali, Banglish, Hindi, Hinglish, Spanish, French, Arabic, and mixed script text.

[![GitHub](https://img.shields.io/badge/GitHub-GodfatherTHR%2Fspamradar-181717?logo=github)](https://github.com/GodfatherTHR/spamradar)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Live%20demo-FFD21E?logo=huggingface&logoColor=black)](https://toriqulhaque-spamradar.static.hf.space/index.html)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](paper/spamradar_paper.pdf)

Author: Md. Toriqul Haque  
Affiliation: University of Scholars  
Email: toriqulhaque27@gmail.com

Live browser demo: https://toriqulhaque-spamradar.static.hf.space/index.html  
Paper: [paper/spamradar_paper.pdf](paper/spamradar_paper.pdf)  
GitHub: https://github.com/GodfatherTHR/spamradar

A SPAM verdict is an automated warning, not proof. Review uncertain messages through an official channel.

## What it catches

- Peer to peer payment traps, including money doubling and fake mistaken transfers
- Fake bKash, Nagad, PayPal, bank, and KYC alerts
- Crypto doubling, fake airdrops, and investment bait
- Delivery, customs fee, and OTP theft scams
- Romance, job, marketplace, loan, and tech support scams
- Obfuscated links such as `bit[.]ly`, `w.a.l.m.a.r.t`, and `hxxps://`
- Hard negatives such as real OTP notices, bank debits, delivery updates, and normal money requests

## Model

The detector is a hybrid of three views:

1. Character and word TF-IDF features, with 180,000 character features and 110,000 word features.
2. 100 engineered signals for URLs, payment requests, OTP language, urgency, script mixing, and scam structure.
3. A frozen multilingual MiniLM encoder with a calibrated linear head.

A validation-fit stacker blends the lexical logit, SVM margin, semantic logit, and heuristic score. The spam threshold is tuned for a false positive rate at or below one percent. Messages between the review and spam thresholds are marked SUSPICIOUS and sent to human review.

The final model uses an INT8 ONNX encoder. The encoder is 118 MB; the FP32 encoder is 470 MB. The repository includes the lexical model, calibration artifacts, and the browser model. The 118 MB encoder is not committed because it exceeds GitHub's per-file limit. Generate it with `python -m src.export_onnx`, or use the Space demo.

## Results

Verified on Windows 11, Python 3.12, CPU only, 16 GB RAM.

| View | PR AUC | ROC AUC | F1 | Precision | Recall | FPR | ECE | Brier | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Realistic 20% prior | 0.9385 | 0.9786 | 0.8315 | 0.9494 | 0.7397 | 0.99% | 0.0039 | 0.0413 | 94.0% |
| Full held-out test | 0.9748 | 0.9773 | 0.8361 | 0.9838 | 0.7269 | 0.99% | 0.0210 | 0.0714 | 87.1% |

- Spam threshold: 0.7068, tuned for FPR at or below 1% on `val_tune`
- Review threshold: 0.3100, tuned for FPR at or below 5%
- Adversarial suite: 97.33% strict pass, 99.33% soft pass, 100% block safe, 0 false blocks
- INT8 encoder: 7.5 ms for batch 1, 32.2 ms for batch 8, 130.0 ms for batch 32
- INT8 cosine similarity to PyTorch embeddings: 0.987

The four remaining strict misses are in the SUSPICIOUS tier. Three are romance or emergency money cases. One is a normal office raffle. No scam is silently missed, and no normal message is auto-blocked.

## Repository layout

```text
.
├── app/
│   ├── api.py                 # FastAPI endpoints and static UI mount
│   └── cli.py                 # interactive terminal
├── src/
│   ├── data_pipeline.py       # ingestion, quality audit, SimHash dedupe, grouped splits
│   ├── features.py            # normalization, language detection, 100 signals
│   ├── synthetic.py           # multilingual templates, hard negatives, obfuscation
│   ├── model_builder.py       # TF-IDF, MiniLM encoder, calibration, thresholds
│   ├── train.py               # training orchestration
│   ├── evaluate.py            # metrics, per-language and per-category results
│   ├── recalibrate.py         # refit stacker and thresholds
│   ├── export_onnx.py         # FP32 and INT8 ONNX export
│   ├── inference.py           # production predictor
│   └── keras_transformer.py   # optional GPU two-stage trainer
├── web/                       # FastAPI browser UI
├── tests/                     # 30 tests and the 150 case adversarial suite
├── scripts/                   # reproducible data download
├── paper/                     # LaTeX source and compiled PDF
├── deploy/
│   ├── static_space/          # free browser demo source and model
│   ├── hf_space/              # Docker and FastAPI deployment files
│   └── cloudflare/            # Worker proxy for a custom subdomain
├── artifacts/                 # lexical model, calibration, metrics, tokenizer
├── config.yaml
└── requirements.txt
```

## Quickstart

```powershell
py -3.12 -m pip install -r requirements.txt

# Build the dataset from the local raw files
py -3.12 -m src.data_pipeline --force

# Full training path
py -3.12 -m src.train

# CPU path used for this build
py -3.12 -m src.train --skip-semantic
py -3.12 -m src.export_onnx
py -3.12 -m src.train --semantic-only --max-semantic-train 32000 --semantic-max-length 64

# Evaluate and recalibrate
py -3.12 -m src.evaluate
py -3.12 -m src.recalibrate

# API and UI
py -3.12 -m uvicorn app.api:app --host 127.0.0.1 --port 8000

# Tests
py -3.12 -m pytest -q
```

The raw datasets are not included in this repository. See `data/processed/data_card.md` for the source list and license notes. The download script expects `KAGGLE_API_TOKEN` in the environment and does not write it to disk.

## API

- `GET /` browser UI
- `POST /predict` single message
- `POST /predict_batch` up to 256 messages
- `GET /health` model backend and thresholds
- `GET /stats` model card, dataset statistics, evaluation metrics
- `GET /examples` demo messages
- `GET /docs` OpenAPI documentation

The response includes `label`, `risk_tier`, `spam_probability`, `language`, `detected_signals`, `model_cues`, `scores`, `latency_ms`, and a `decision` object with `source` and `rule_reasons`.

## Browser demo

The free static demo at https://toriqulhaque-spamradar.static.hf.space/index.html runs the lexical and signal branch in the browser through Pyodide. It downloads a 6.5 MB compressed model, needs no server, and does not send message text anywhere. The full semantic encoder is not loaded in that demo because it would add 118 MB to the first page load.

The source for the static demo is under `deploy/static_space`. The Cloudflare Worker under `deploy/cloudflare` can put it on a custom subdomain.

## Limitations

- Public SMS corpora contain label noise. The quality audit reduces it but does not remove it.
- A text only model cannot resolve every romance or emergency money case. Use SUSPICIOUS as a review queue.
- The synthetic templates add modern scam families, but they can make the task easier than real traffic.
- The semantic encoder is frozen. A fully fine tuned multilingual transformer may improve paraphrase handling but needs a GPU.
- Scams change. Retrain, add hard negatives, and rerun the adversarial suite regularly.

This system is for defensive filtering and research. Do not use it to surveil people or block legitimate communication without human review and an appeal path.

## Citation

```bibtex
@misc{haque2026spamradar,
  title = {Catching Multilingual Spam on a CPU: A Hybrid Detector with Review Tiers},
  author = {Haque, Md. Toriqul},
  year = {2026},
  institution = {University of Scholars},
  url = {https://github.com/GodfatherTHR/spamradar}
}
```

## License

No license has been selected yet. Contact the author before reuse.
