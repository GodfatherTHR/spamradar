---
title: SpamRadar
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
---

# SpamRadar Browser Demo

Multilingual spam and social engineering detection that runs entirely in the browser. No server, no API key, and no message upload.

The free demo uses the trained lexical and signal branch: character and word TF-IDF features, 100 engineered scam signals, the high confidence escalation rules, and a precision constrained threshold pair. The full FastAPI bundle in the project also includes the frozen multilingual MiniLM encoder and the calibrated stacker. Hugging Face currently requires a PRO subscription to create free Docker Spaces, so the hosted demo uses the browser build instead.

## Verified browser branch results

On the held out realistic test set with a 20 percent spam prior:

- PR AUC: 0.9380
- F1: 0.8301
- Precision: 95.2 percent
- Recall: 73.6 percent
- False positive rate: 0.93 percent
- Brier score: 0.0555

The spam threshold is 0.9245 and the review threshold is 0.6372. Both were tuned on a disjoint validation split, with the spam threshold constrained to a false positive rate at or below 1 percent.

## Full model

The full hybrid model adds a multilingual MiniLM encoder and a calibrated stacker. On the same held out test it reaches a PR AUC of 0.9385, an F1 of 0.8315, a precision of 94.9 percent, a recall of 74.0 percent, a false positive rate of 0.99 percent, and an ECE of 0.0039. The Docker and FastAPI bundle is included in the project code release, ready for a free container host such as Render or Koyeb, or a Hugging Face PRO Space.

## Privacy

All inference happens in the browser. Message text is not sent to a server. The first load downloads the Python runtime and the detector, then the browser cache stores them. Do not send secrets or live OTP codes to any public demo.

## Limits

This is a defensive text filter. A SPAM label is a warning, not proof. Review SUSPICIOUS messages through an official channel. The browser demo is slower than the 15 to 21 ms native CPU path because it runs Python in WebAssembly.
