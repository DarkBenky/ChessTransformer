import tensorflow as tf
from tensorflow import keras
from io import StringIO
import os
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Embedding,
    GlobalAveragePooling1D,
)
from dataHelper import parse_board_response
import numpy as np
import requests
import wandb

STEPS = 1_000_000_000_000
BATCH_SIZE = 32
API_URL = "http://localhost:1323/getData"
NUM_TOKENS = 15
SEQ_LEN = 65
MODEL_DIM = 64


def transformer_block(x, heads=4, ff_dim=128, dropout=0.1):
    attn_out = MultiHeadAttention(num_heads=heads, key_dim=MODEL_DIM // heads)(x, x)
    attn_out = Dropout(dropout)(attn_out)
    x = LayerNormalization(epsilon=1e-6)(x + attn_out)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(MODEL_DIM)(ff)
    ff = Dropout(dropout)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def build_eval_model() -> keras.Model:
    inp = Input(shape=(SEQ_LEN,), dtype="int32", name="board_tokens")

    tok_emb = Embedding(input_dim=NUM_TOKENS, output_dim=MODEL_DIM)(inp)
    pos_ids = tf.expand_dims(tf.range(start=0, limit=SEQ_LEN, delta=1), axis=0)
    pos_emb = Embedding(input_dim=SEQ_LEN, output_dim=MODEL_DIM)(pos_ids)
    x = tok_emb + pos_emb

    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    out = Dense(1, name="eval_out")(x)
    model = keras.Model(inp, out, name="eval_model")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_next_board_model() -> keras.Model:
    inp = Input(shape=(SEQ_LEN,), dtype="int32", name="board_tokens")

    tok_emb = Embedding(input_dim=NUM_TOKENS, output_dim=MODEL_DIM)(inp)
    pos_ids = tf.expand_dims(tf.range(start=0, limit=SEQ_LEN, delta=1), axis=0)
    pos_emb = Embedding(input_dim=SEQ_LEN, output_dim=MODEL_DIM)(pos_ids)
    x = tok_emb + pos_emb

    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)
    x = transformer_block(x, heads=8, ff_dim=512, dropout=0.1)

    out = Dense(NUM_TOKENS, name="next_board_logits")(x)
    model = keras.Model(inp, out, name="next_board_model")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="token_acc")],
    )
    return model


def fetch_batch(batch_size: int):
    x_batch, y_board_batch, y_eval_batch = [], [], []
    failures = 0

    while len(x_batch) < batch_size:
        try:
            resp = requests.get(API_URL, timeout=10)
            resp.raise_for_status()
            parsed = parse_board_response(resp.json())
            x_batch.append(parsed["X"].astype(np.int32))
            y_board_batch.append(parsed["Y"].astype(np.int32))
            y_eval_batch.append(np.float32(parsed["eval"]))
        except Exception:
            failures += 1
            if failures > batch_size * 5:
                break

    if len(x_batch) == 0:
        return None, None, None

    x_np = np.stack(x_batch, axis=0)
    y_board_np = np.stack(y_board_batch, axis=0)
    y_eval_np = np.array(y_eval_batch, dtype=np.float32)
    return x_np, y_board_np, y_eval_np


def get_model_summary_text(model: keras.Model) -> str:
    buf = StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()


if __name__ == "__main__":
    wandb.init(
        project="chess-transformer",
        config={"steps": STEPS, "batch_size": BATCH_SIZE, "lr": 1e-4},
    )

    eval_model = build_eval_model()
    next_board_model = build_next_board_model()

    eval_model.summary()
    next_board_model.summary()

    # Log model summaries to W&B
    eval_summary_text = get_model_summary_text(eval_model)
    next_summary_text = get_model_summary_text(next_board_model)

    with open("eval_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(eval_summary_text)
    with open("next_board_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(next_summary_text)

    wandb.save("eval_model_summary.txt")
    wandb.save("next_board_model_summary.txt")
    wandb.run.summary["eval_model_params"] = int(eval_model.count_params())
    wandb.run.summary["next_board_model_params"] = int(next_board_model.count_params())

    os.makedirs("checkpoints", exist_ok=True)
    best_eval_loss = float("inf")
    best_next_loss = float("inf")

    for step in range(STEPS):
        x_batch, y_board_batch, y_eval_batch = fetch_batch(BATCH_SIZE)
        if x_batch is None:
            print(f"Step {step + 1}/{STEPS}: no batch fetched")
            continue

        x_in = x_batch

        eval_metrics = eval_model.train_on_batch(x_in, y_eval_batch, return_dict=True)
        next_metrics = next_board_model.train_on_batch(x_in, y_board_batch, return_dict=True)

        wandb.log(
            {
                "step": step + 1,
                "eval/loss": float(eval_metrics["loss"]),
                "eval/mae": float(eval_metrics["mae"]),
                "next_board/loss": float(next_metrics["loss"]),
                "next_board/token_acc": float(next_metrics["token_acc"]),
                "batch_size": int(x_batch.shape[0]),
            },
            step=step + 1,
        )

        # Save best checkpoints
        if float(eval_metrics["loss"]) < best_eval_loss:
            best_eval_loss = float(eval_metrics["loss"])
            eval_model.save("checkpoints/best_eval_model.keras")
            wandb.run.summary["best_eval_loss"] = best_eval_loss
            wandb.run.summary["best_eval_step"] = step + 1

        if float(next_metrics["loss"]) < best_next_loss:
            best_next_loss = float(next_metrics["loss"])
            next_board_model.save("checkpoints/best_next_board_model.keras")
            wandb.run.summary["best_next_board_loss"] = best_next_loss
            wandb.run.summary["best_next_board_step"] = step + 1

        print(
            f"Step {step + 1}/{STEPS} | "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_mae={eval_metrics['mae']:.4f} | "
            f"next_loss={next_metrics['loss']:.4f} "
            f"next_acc={next_metrics['token_acc']:.4f}"
        )

    wandb.finish()
