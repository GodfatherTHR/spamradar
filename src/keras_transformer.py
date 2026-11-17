from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

from .utils import ensure_dir, load_config, resolve_path, set_seed


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


class FocalLoss(keras.losses.Loss):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        from_logits: bool = True,
        name: str = "focal_loss",
    ):
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        y_true = keras.ops.reshape(keras.ops.cast(y_true, "int32"), (-1,))
        y_true = keras.ops.one_hot(y_true, 2)
        y_pred = keras.ops.cast(y_pred, "float32")
        if self.from_logits:
            y_pred = keras.ops.softmax(y_pred, axis=-1)
        y_pred = keras.ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / 2.0
        ce = -keras.ops.sum(y_true * keras.ops.log(y_pred), axis=-1)
        p_t = keras.ops.sum(y_true * y_pred, axis=-1)
        loss = self.alpha * keras.ops.power(1.0 - p_t, self.gamma) * ce
        return keras.ops.mean(loss)

    def get_config(self):
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
            "name": self.name,
        }


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        warmup = keras.ops.cast(self.warmup_steps, "float32")
        total = keras.ops.cast(self.total_steps, "float32")
        warmup_lr = self.base_lr * (step / warmup)
        progress = (step - warmup) / keras.ops.maximum(total - warmup, 1.0)
        cosine = 0.5 * (1.0 + keras.ops.cos(math.pi * progress))
        cosine_lr = self.min_lr + (self.base_lr - self.min_lr) * cosine
        return keras.ops.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


def model_classes(preset: str):
    if preset.startswith("deberta"):
        return (
            keras_hub.models.DebertaV3TextClassifier,
            keras_hub.models.DebertaV3TextClassifierPreprocessor,
        )
    if preset.startswith("distil_bert"):
        return (
            keras_hub.models.DistilBertTextClassifier,
            keras_hub.models.DistilBertTextClassifierPreprocessor,
        )
    if preset.startswith("bert"):
        return (
            keras_hub.models.BertTextClassifier,
            keras_hub.models.BertTextClassifierPreprocessor,
        )
    raise ValueError(
        "Unsupported preset. Use a KerasHub multilingual preset such as "
        "deberta_v3_base_multi or distil_bert_base_multi."
    )


def build_model(preset: str, sequence_length: int):
    classifier_cls, preprocessor_cls = model_classes(preset)
    preprocessor = preprocessor_cls.from_preset(preset, sequence_length=int(sequence_length))
    classifier = classifier_cls.from_preset(
        preset, num_classes=2, preprocessor=None
    )
    classifier.backbone.trainable = False
    return classifier, preprocessor


def tokenize_eagerly(preprocessor, texts):
    log(f"Tokenizing {len(texts):,} texts eagerly (sequence_length bounded on the preprocessor)...")
    return preprocessor([str(t) for t in texts])


def _transformer_blocks(backbone):
    for layer in backbone.layers:
        sublayers = getattr(layer, "layers", None)
        if sublayers and len(sublayers) >= 4:
            named = [sub for sub in sublayers if "layer" in sub.name.lower() or "block" in sub.name.lower()]
            if len(named) >= 4:
                return named
    named = [layer for layer in backbone.layers if "layer" in layer.name.lower()]
    return named if len(named) >= 4 else []


def unfreeze_top_layers(backbone, n: int = 4) -> dict:
    blocks = _transformer_blocks(backbone)
    if not blocks:
        backbone.trainable = True
        return {"strategy": "full_backbone", "unfrozen_blocks": None}
    backbone.trainable = True
    for layer in blocks[:-n]:
        layer.trainable = False
    return {
        "strategy": "top_encoder_layers",
        "unfrozen_blocks": int(min(n, len(blocks))),
        "total_blocks": int(len(blocks)),
    }


def count_trainable(model) -> tuple[int, int]:
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    non_trainable = int(sum(int(np.prod(w.shape)) for w in model.non_trainable_weights))
    return trainable, non_trainable


def load_split(processed_dir: Path, name: str, limit: int | None = None):
    df = pd.read_csv(processed_dir / f"{name}.csv")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    if limit:
        df = df.sample(frac=1.0, random_state=42).groupby("label", group_keys=False).head(
            max(1, limit // 2)
        ).reset_index(drop=True)
    return df


def compile_classifier(classifier, learning_rate, focal_alpha=0.25, label_smoothing=0.05, clipnorm=1.0, gradient_accumulation_steps=1):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.01,
        clipnorm=clipnorm,
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )
    classifier.compile(
        optimizer=optimizer,
        loss=FocalLoss(alpha=focal_alpha, label_smoothing=label_smoothing, from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.AUC(curve="PR", name="pr_auc", from_logits=True),
        ],
    )
    return classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage multilingual transformer fine-tuning with KerasHub."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--preset", default="deberta_v3_base_multi")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=4)
    parser.add_argument("--unfreeze-layers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    set_seed(seed)
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    artifacts_dir = ensure_dir(config["paths"].get("artifacts_dir", "artifacts"))

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
        log(f"GPU detected: {len(gpus)} device(s); mixed_float16 enabled")
    else:
        keras.mixed_precision.set_global_policy("float32")
        log("CPU-only training; float32 enabled")

    train_df = load_split(processed_dir, "train", args.max_train)
    val_df = load_split(processed_dir, "val", args.max_val)
    if args.smoke_test:
        train_df = train_df.groupby("label", group_keys=False).head(8).reset_index(drop=True)
        val_df = val_df.groupby("label", group_keys=False).head(4).reset_index(drop=True)
        args.stage1_epochs = 1
        args.stage2_epochs = 1
        args.sequence_length = min(args.sequence_length, 64)
        args.batch_size = min(args.batch_size, 2)
    log(f"Train rows={len(train_df):,}, val rows={len(val_df):,}")

    log(f"Loading KerasHub preset: {args.preset}")
    classifier, preprocessor = build_model(args.preset, args.sequence_length)
    train_inputs = tokenize_eagerly(preprocessor, train_df["text"].tolist())
    val_inputs = tokenize_eagerly(preprocessor, val_df["text"].tolist())
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    y_val = val_df["label"].to_numpy(dtype=np.int32)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=2, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(artifacts_dir / "keras_best.weights.h5"),
            monitor="val_pr_auc", mode="max", save_best_only=True, save_weights_only=True,
        ),
        keras.callbacks.CSVLogger(str(artifacts_dir / "keras_training_log.csv"), append=True),
        keras.callbacks.TerminateOnNaN(),
    ]

    log("Stage 1: frozen multilingual backbone, head warmup")
    classifier = compile_classifier(
        classifier,
        learning_rate=1e-3,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainable, frozen = count_trainable(classifier)
    log(f"  trainable={trainable:,}, frozen={frozen:,}")
    t0 = time.time()
    classifier.fit(
        x=train_inputs,
        y=y_train,
        validation_data=(val_inputs, y_val),
        batch_size=args.batch_size,
        epochs=args.stage1_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    log(f"Stage 1 finished in {time.time() - t0:.1f}s")

    log("Stage 2: partial unfreeze with layer-wise schedule")
    info = unfreeze_top_layers(classifier.backbone, args.unfreeze_layers)
    log(f"  unfreeze strategy: {info}")
    steps_per_epoch = max(
        1,
        math.ceil(len(train_df) / args.batch_size / max(1, args.gradient_accumulation_steps)),
    )
    total_steps = steps_per_epoch * max(1, args.stage2_epochs)
    schedule = WarmupCosineDecay(
        base_lr=2e-5,
        warmup_steps=max(10, int(total_steps * 0.1)),
        total_steps=total_steps,
    )
    classifier = compile_classifier(
        classifier,
        learning_rate=schedule,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainable, frozen = count_trainable(classifier)
    log(f"  trainable={trainable:,}, frozen={frozen:,}, update_steps={total_steps}")
    t0 = time.time()
    classifier.fit(
        x=train_inputs,
        y=y_train,
        validation_data=(val_inputs, y_val),
        batch_size=args.batch_size,
        epochs=args.stage2_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    log(f"Stage 2 finished in {time.time() - t0:.1f}s")

    weights_path = artifacts_dir / "keras_multilingual_classifier.weights.h5"
    classifier.save_weights(str(weights_path))
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "unfreeze": info,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "weights_path": str(weights_path),
        "mixed_precision": keras.mixed_precision.global_policy().name,
        "smoke_test": bool(args.smoke_test),
    }
    with open(artifacts_dir / "keras_model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    log(f"Saved weights to {weights_path}")
    log("Done. The production CPU branch remains the calibrated hybrid model.")


if __name__ == "__main__":
    main()
