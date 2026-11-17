# Evaluation report – Multilingual Spam & Scam Detection

Generated: `2026-11-16T22:28:46.953334+00:00`

## Model

- Type: **hybrid_tfidf_signals_plus_multilingual_minilm**
- Multilingual backbone: **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**
- Spam threshold: **0.706775** (FPR target ≤ 0.01)
- Review threshold: **0.31**

## Held-out test metrics

### Realistic 20% spam prior

| Rows | Spam prior | PR-AUC | ROC-AUC | F1 | Precision | Recall | FPR | Brier | ECE | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10392 | 0.2000 | 0.9385 | 0.9786 | 0.8315 | 0.9494 | 0.7397 | 0.0099 | 0.0413 | 0.0039 | 0.9401 |

Confusion matrix (rows=actual, cols=predicted): `[[8232, 82], [541, 1537]]`

### Full held-out test

| Rows | Spam prior | PR-AUC | ROC-AUC | F1 | Precision | Recall | FPR | Brier | ECE | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15162 | 0.4517 | 0.9748 | 0.9773 | 0.8361 | 0.9838 | 0.7269 | 0.0099 | 0.0714 | 0.0210 | 0.8713 |

Confusion matrix (rows=actual, cols=predicted): `[[8232, 82], [1870, 4978]]`

## Per-language breakdown

| Language | Rows | Spam prior | Precision | Recall | F1 | FPR | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| English | 8609 | 0.1917 | 0.9374 | 0.7079 | 0.8066 | 0.0112 | 0.9203 |
| Hinglish | 371 | 0.3531 | 0.9921 | 0.9542 | 0.9728 | 0.0042 | 0.9977 |
| French | 357 | 0.1204 | 1.0000 | 0.8605 | 0.9250 | 0.0000 | 0.9940 |
| Hindi | 231 | 0.1342 | 1.0000 | 0.4839 | 0.6522 | 0.0000 | 0.9550 |
| Bengali | 167 | 0.4251 | 1.0000 | 0.9155 | 0.9559 | 0.0000 | 0.9977 |
| Other / rare | 166 | 0.0663 | 1.0000 | 0.6364 | 0.7778 | 0.0000 | 0.9001 |
| Banglish | 161 | 0.4348 | 0.9706 | 0.9429 | 0.9565 | 0.0220 | 0.9984 |
| Spanish | 150 | 0.1600 | 1.0000 | 0.5000 | 0.6667 | 0.0000 | 0.9684 |
| mixed | 113 | 0.3186 | 0.9714 | 0.9444 | 0.9577 | 0.0130 | 0.9970 |
| Arabic | 67 | 0.1642 | 1.0000 | 0.7273 | 0.8421 | 0.0000 | 0.9842 |

## Per-category breakdown

| Category | Rows | Spam prior | Precision | Recall | F1 | FPR | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hard negative · personal / benign | 7138 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0090 | None |
| Scam · other | 1350 | 1.0000 | 1.0000 | 0.6526 | 0.7898 | 0.0000 | None |
| Hard negative · OTP / bank / service | 958 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0188 | None |
| Scam / spam · lottery / promotional | 475 | 1.0000 | 1.0000 | 0.9200 | 0.9583 | 0.0000 | None |
| Hard negative · other | 112 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | None |
| Scam · bank / KYC / UPI / investment | 106 | 1.0000 | 1.0000 | 0.9811 | 0.9905 | 0.0000 | None |
| Scam · P2P / crypto / marketplace / romance | 79 | 1.0000 | 1.0000 | 0.7848 | 0.8794 | 0.0000 | None |
| Hard negative · delivery | 70 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | None |
| Scam · delivery / OTP | 40 | 1.0000 | 1.0000 | 0.7500 | 0.8571 | 0.0000 | None |
| Scam · job | 24 | 1.0000 | 1.0000 | 0.8750 | 0.9333 | 0.0000 | None |
| Hard negative · opt-in promo | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | None |
| Hard negative · marketplace | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | None |
| Scam · phishing | 4 | 1.0000 | 1.0000 | 0.5000 | 0.6667 | 0.0000 | None |

## Adversarial suite

- Cases: **150**
- Hard pass rate: **97.33%**
- Soft pass rate (review counts for scam): **99.33%**
- Block-safe rate (no normal message auto-blocked): **100.00%**
- False blocks: **0**
- False reviews of normal messages: **1**
- Missed scams (predicted NORMAL): **0**

### Adversarial failures

- **en_romance_doctor** [SPAM → SUSPICIOUS, p=0.358, English] I feel a deep connection with you. Can you help me with $800 for my mother's hospital bill? I will repay you when I arrive.
- **en_romance_ticket** [SPAM → SUSPICIOUS, p=0.375, English] I bought a ticket to visit you, but customs is asking for a $250 clearance fee. Please pay it so I can finally see you.
- **bn_romance** [SPAM → SUSPICIOUS, p=0.482, Bengali] প্রিয়, আমি বিদেশে সেনাবাহিনীতে আছি। তোমার সাথে দেখা করতে ২০,০০০ টাকা দরকার। বিকাশে পাঠাও।
- **ham_office_gift** [NORMAL → SUSPICIOUS, p=0.651, English] You won the office raffle! Come pick up your gift card from reception whenever you are free.

## Plots

- `artifacts/plots/confusion_matrix.png`
- `artifacts/plots/pr_curve.png`
- `artifacts/plots/roc_curve.png`
- `artifacts/plots/score_distribution.png`
- `artifacts/plots/calibration_curve.png`
- `artifacts/plots/per_language_f1.png`

## Notes and limitations

- The headline operating point is precision-constrained: the spam threshold is tuned for FPR ≤ 1% on a disjoint validation split.
- The full test set has a ~45% spam prior because it is built from merged spam/ham corpora. The realistic subset resamples it to 20% for a more deployment-like view.
- SMS datasets contain label noise. Obvious contradictory noisy-source rows were filtered and ambiguous dedupe clusters were dropped, but some noise remains.
- Text-only detection cannot resolve every ambiguous case (for example, a genuine "I sent money by mistake, please return it" versus the same message sent by a scammer). Use the SUSPICIOUS tier for review.
- The legacy English-only DeBERTa-v3-extra-small H5 model is not used for multilingual inference.
