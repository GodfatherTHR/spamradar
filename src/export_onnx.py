from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .utils import ensure_dir, load_config, resolve_path, set_seed


def log(message: str) -> None:
    print(message, flush=True)


class MeanPoolingEncoder(torch.nn.Module):
    def __init__(self, model, use_token_type_ids: bool = True):
        super().__init__()
        self.model = model
        self.use_token_type_ids = bool(use_token_type_ids)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.use_token_type_ids and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.model(**kwargs)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1)


def encode_torch(wrapper, tokenizer, texts, max_length=128):
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    names = ["input_ids", "attention_mask"]
    if wrapper.use_token_type_ids and "token_type_ids" in enc:
        names.append("token_type_ids")
    args = [enc[name] for name in names]
    with torch.no_grad():
        out = wrapper(*args)
    return out.cpu().numpy(), enc, names


def validate_onnx(session, enc, names, reference, label):
    feed = {name: enc[name].numpy().astype(np.int64) for name in names if name in enc}
    output = session.run(None, feed)[0]
    cosine = float(
        np.mean(
            np.sum(output * reference, axis=1)
            / (np.linalg.norm(output, axis=1) * np.linalg.norm(reference, axis=1) + 1e-9)
        )
    )
    log(f"  {label} cosine similarity vs PyTorch: {cosine:.6f}")
    return output, cosine


def benchmark_session(session, tokenizer, names, texts, batches=(1, 8, 32), runs=12):
    results = {}
    for batch in batches:
        sample = (texts * ((batch // len(texts)) + 1))[:batch]
        enc = tokenizer(
            sample, padding=True, truncation=True, max_length=128, return_tensors="np"
        )
        feed = {name: enc[name].astype(np.int64) for name in names if name in enc}
        for _ in range(2):
            session.run(None, feed)
        start = time.perf_counter()
        for _ in range(runs):
            session.run(None, feed)
        elapsed = (time.perf_counter() - start) / runs
        results[f"batch_{batch}"] = round(elapsed * 1000.0, 3)
    return results


def benchmark_torch(wrapper, tokenizer, texts, batches=(1, 8, 32), runs=12):
    results = {}
    for batch in batches:
        sample = (texts * ((batch // len(texts)) + 1))[:batch]
        enc = tokenizer(
            sample, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        names = ["input_ids", "attention_mask"]
        if wrapper.use_token_type_ids and "token_type_ids" in enc:
            names.append("token_type_ids")
        args = [enc[name] for name in names]
        with torch.no_grad():
            for _ in range(2):
                wrapper(*args)
            start = time.perf_counter()
            for _ in range(runs):
                wrapper(*args)
            elapsed = (time.perf_counter() - start) / runs
        results[f"batch_{batch}"] = round(elapsed * 1000.0, 3)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the multilingual encoder to quantized ONNX.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    artifacts_dir = ensure_dir(
        args.output_dir or resolve_path(config["paths"]["artifacts_dir"])
    )
    metadata_path = artifacts_dir / "semantic_metadata.json"
    model_name = args.model
    if not model_name and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            model_name = json.load(fh).get("model_name")
    model_name = model_name or config["model"]["semantic"]["backbone"]
    max_length = int(config["model"]["semantic"].get("max_seq_length", 128))
    log(f"Exporting multilingual encoder: {model_name}")

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    accepts_token_types = "token_type_ids" in inspect.signature(model.forward).parameters
    wrapper = MeanPoolingEncoder(model, use_token_type_ids=accepts_token_types)
    wrapper.eval()

    tokenizer_dir = artifacts_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    log(f"Saved tokenizer to {tokenizer_dir}")

    sample_texts = [
        "Give me five dollars and I will give you an RTX 3090.",
        "Your account is suspended. Verify immediately at this link.",
        "আপনি পাঁচ লাখ টাকা জিতেছেন। পুরস্কার পেতে যোগাযোগ করুন।",
        "Aapka KYC pending hai, turant link par click karein.",
        "Your OTP is 482913. Do not share it with anyone.",
        "Bro can you send me five dollars for lunch?",
    ]
    reference, enc, names = encode_torch(wrapper, tokenizer, sample_texts, max_length=max_length)
    input_names = list(names)
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    fp32_path = artifacts_dir / "semantic_encoder.onnx"
    dummy_args = tuple(enc[name] for name in input_names)
    log(f"Exporting FP32 ONNX to {fp32_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_args,
            str(fp32_path),
            input_names=input_names,
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=int(args.opset),
            do_constant_folding=True,
            dynamo=False,
        )

    import onnxruntime as ort

    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    _, cosine_fp32 = validate_onnx(fp32_session, enc, input_names, reference, "FP32")

    st_reference = None
    st_cosine_fp32 = None
    st_cosine_int8 = None
    try:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(model_name, device="cpu")
        st_model.max_seq_length = max_length
        st_reference = np.asarray(
            st_model.encode(sample_texts, normalize_embeddings=True), dtype=np.float32
        )
        feed = {name: enc[name].numpy().astype(np.int64) for name in input_names if name in enc}
        fp32_out = fp32_session.run(None, feed)[0]
        st_cosine_fp32 = float(
            np.mean(
                np.sum(fp32_out * st_reference, axis=1)
                / (np.linalg.norm(fp32_out, axis=1) * np.linalg.norm(st_reference, axis=1) + 1e-9)
            )
        )
        log(f"  FP32 cosine vs SentenceTransformer.encode: {st_cosine_fp32:.6f}")
    except Exception as exc:
        log(f"  SentenceTransformer cross-check skipped: {type(exc).__name__}: {exc}")

    int8_path = artifacts_dir / "semantic_encoder_int8.onnx"
    cosine_int8 = None
    if not args.skip_quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        log(f"Quantizing dynamically to INT8: {int8_path}")
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        _, cosine_int8 = validate_onnx(int8_session, enc, input_names, reference, "INT8")
        if st_reference is not None:
            feed = {name: enc[name].numpy().astype(np.int64) for name in input_names if name in enc}
            int8_out = int8_session.run(None, feed)[0]
            st_cosine_int8 = float(
                np.mean(
                    np.sum(int8_out * st_reference, axis=1)
                    / (np.linalg.norm(int8_out, axis=1) * np.linalg.norm(st_reference, axis=1) + 1e-9)
                )
            )
            log(f"  INT8 cosine vs SentenceTransformer.encode: {st_cosine_int8:.6f}")

    benchmark = {
        "model": model_name,
        "max_seq_length": max_length,
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path) if int8_path.exists() else None,
        "cosine_similarity_fp32": round(cosine_fp32, 6),
        "cosine_similarity_int8": round(cosine_int8, 6) if cosine_int8 is not None else None,
        "cosine_vs_sentence_transformers_fp32": round(st_cosine_fp32, 6) if st_cosine_fp32 is not None else None,
        "cosine_vs_sentence_transformers_int8": round(st_cosine_int8, 6) if st_cosine_int8 is not None else None,
    }
    if not args.skip_benchmark:
        log("Benchmarking CPU latency...")
        benchmark["latency_ms"] = {
            "pytorch": benchmark_torch(wrapper, tokenizer, sample_texts),
            "onnx_fp32": benchmark_session(fp32_session, tokenizer, input_names, sample_texts),
        }
        if not args.skip_quantize and int8_path.exists():
            benchmark["latency_ms"]["onnx_int8"] = benchmark_session(
                ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"]),
                tokenizer, input_names, sample_texts,
            )
        log(json.dumps(benchmark["latency_ms"], indent=2))

    with open(artifacts_dir / "onnx_benchmark.json", "w", encoding="utf-8") as fh:
        json.dump(benchmark, fh, ensure_ascii=False, indent=2)
    log(f"Wrote {artifacts_dir / 'onnx_benchmark.json'}")
    log("Done.")


if __name__ == "__main__":
    main()
