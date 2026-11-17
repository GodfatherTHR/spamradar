---
title: SpamRadar
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SpamRadar

Multilingual spam and social engineering detection with a calibrated review tier and a fast CPU inference path.

This Space runs the FastAPI service and the browser UI. The detector combines character and word TF-IDF features, 100 engineered scam signals, a frozen multilingual MiniLM encoder, a calibrated probability stacker, and a small set of high confidence escalation rules. The encoder is exported to INT8 ONNX.

## Endpoints

- `GET /` browser UI
- `POST /predict` single message
- `POST /predict_batch` up to 256 messages
- `GET /health` model backend and thresholds
- `GET /stats` model and evaluation data
- `GET /examples` demo messages
- `GET /docs` OpenAPI documentation

Example:

```bash
curl -X POST "$SPACE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Give me $5 bucks I will give you an RTX 3090"}'
```

## Verified results

On a held out test set with a 20 percent spam prior, the model reached a PR AUC of 0.9385, an F1 of 0.8315, a precision of 94.9 percent, a recall of 74.0 percent, and a false positive rate of 0.99 percent. On a 150 case adversarial suite it reached a 97.33 percent strict pass rate, a 99.33 percent soft pass rate, and a 100 percent block safe rate, with no false blocks and no missed scams.

The INT8 encoder takes about 7.5 ms for one message. A warmed single message API call takes about 15 to 21 ms on a four core Intel i3 CPU.

## Privacy

The Space does not store message text. Predictions are computed in memory. Do not send secrets or live OTP codes to a public demo.

If you put a Cloudflare Worker in front of the Space, set the Space secret `SPAMRADAR_API_KEY` and the Worker secret of the same name. The Worker adds the `X-API-Key` header, and the API rejects prediction calls that do not carry it. If the variable is not set, the public demo works without a key.

## Model limits

This is a text only defensive filter. A SPAM label is a warning, not proof. Review SUSPICIOUS messages through an official channel. The free Space can sleep after inactivity, so the first request after a pause may be slow.
