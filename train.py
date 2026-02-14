import tensorflow as tf
from tensorflow import keras
from io import StringIO
from io import BytesIO
import os
import shutil
from datetime import datetime
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Embedding,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
)
from dataHelper import parse_board_response
import numpy as np
import requests
import wandb

# Prevent TensorFlow from pre-allocating all GPU memory.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

try:
    import chess

    CHESS_AVAILABLE = True
except Exception:
    CHESS_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import cairosvg

    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

STEPS = 1_000_000_000_000
BATCH_SIZE = 32
API_URL = "http://localhost:1323/getData"
NUM_TOKENS = 15
SEQ_LEN = 65
MODEL_DIM = 64
NUM_HEADS = 4
FF_DIM = 256
NUM_TRANSFORMER_BLOCKS = 4
DROPOUT_RATE = 0.2
SELF_PLAY_GIF_EVERY_STEPS = 100
SELF_PLAY_PLIES = 60  # 30 moves per side
SELF_PLAY_ALWAYS_FROM_START = True
SELF_PLAY_USE_DROPOUT_INFERENCE = True
SELF_PLAY_MC_DROPOUT_SAMPLES = 4
SELF_PLAY_TEMPERATURE = 0.5
SELF_PLAY_TOP_K = 4
EVAL_INITIAL_LR = 2e-4
NEXT_INITIAL_LR = 2e-4
LR_REDUCE_PATIENCE_STEPS = 1000
LR_REDUCE_FACTOR = 0.9
LEGAL_LOSS_WEIGHT = 0.2
ERROR_RATE_EPS = 1e-3
ERROR_RATE_MIN_ABS_TARGET = 0.1
EVAL_PERSPECTIVE = "white"  # "white" or "side_to_move"
EVAL_TANH_K_PAWNS = 4.0
VALIDATE_EVERY_STEPS = 100
VAL_BATCH_SIZE = 256
VAL_CACHE_PATH = "artifacts/validation/val_batch.npz"
RES_TOWER_CHANNELS = 320
RES_TOWER_BLOCKS = 8
MODEL_ROOT_DIR = "models"
EVAL_MODEL_TAG = "eval_cnn"
NEXT_BOARD_ARCH = "cnn"  # "cnn" or "transformer"
NEXT_BOARD_MODEL_TAG = f"next_board_{NEXT_BOARD_ARCH}"
BEST_EVAL_CKPT_PATH = f"{MODEL_ROOT_DIR}/{EVAL_MODEL_TAG}/best_eval_model.keras"
BEST_NEXT_CKPT_PATH = f"{MODEL_ROOT_DIR}/{NEXT_BOARD_MODEL_TAG}/best_next_board_{NEXT_BOARD_ARCH}.keras"
LEGACY_BEST_EVAL_CKPT_PATH = "checkpoints/best_eval_model.keras"
LEGACY_BEST_NEXT_CKPT_PATH = "checkpoints/best_next_board_model.keras"
RESUME_FROM_BEST_CHECKPOINT = True
RESUME_EVAL_MODEL = True
RESUME_NEXT_BOARD_MODEL = False

EMPTY_SQUARE = 0
WHITE_PAWN = 1
WHITE_KNIGHT = 2
WHITE_BISHOP = 3
WHITE_ROOK = 4
WHITE_QUEEN = 5
WHITE_KING = 6
BLACK_PAWN = 7
BLACK_KNIGHT = 8
BLACK_BISHOP = 9
BLACK_ROOK = 10
BLACK_QUEEN = 11
BLACK_KING = 12
WHITE_TO_MOVE = 13
BLACK_TO_MOVE = 14

TOKEN_TO_CHESS_PIECE = {
    WHITE_PAWN: chess.Piece(chess.PAWN, chess.WHITE) if CHESS_AVAILABLE else None,
    WHITE_KNIGHT: chess.Piece(chess.KNIGHT, chess.WHITE) if CHESS_AVAILABLE else None,
    WHITE_BISHOP: chess.Piece(chess.BISHOP, chess.WHITE) if CHESS_AVAILABLE else None,
    WHITE_ROOK: chess.Piece(chess.ROOK, chess.WHITE) if CHESS_AVAILABLE else None,
    WHITE_QUEEN: chess.Piece(chess.QUEEN, chess.WHITE) if CHESS_AVAILABLE else None,
    WHITE_KING: chess.Piece(chess.KING, chess.WHITE) if CHESS_AVAILABLE else None,
    BLACK_PAWN: chess.Piece(chess.PAWN, chess.BLACK) if CHESS_AVAILABLE else None,
    BLACK_KNIGHT: chess.Piece(chess.KNIGHT, chess.BLACK) if CHESS_AVAILABLE else None,
    BLACK_BISHOP: chess.Piece(chess.BISHOP, chess.BLACK) if CHESS_AVAILABLE else None,
    BLACK_ROOK: chess.Piece(chess.ROOK, chess.BLACK) if CHESS_AVAILABLE else None,
    BLACK_QUEEN: chess.Piece(chess.QUEEN, chess.BLACK) if CHESS_AVAILABLE else None,
    BLACK_KING: chess.Piece(chess.KING, chess.BLACK) if CHESS_AVAILABLE else None,
}

CHESS_PIECE_TO_TOKEN = {
    "P": WHITE_PAWN,
    "N": WHITE_KNIGHT,
    "B": WHITE_BISHOP,
    "R": WHITE_ROOK,
    "Q": WHITE_QUEEN,
    "K": WHITE_KING,
    "p": BLACK_PAWN,
    "n": BLACK_KNIGHT,
    "b": BLACK_BISHOP,
    "r": BLACK_ROOK,
    "q": BLACK_QUEEN,
    "k": BLACK_KING,
}

PIECE_TO_GLYPH = {
    EMPTY_SQUARE: "",
    WHITE_PAWN: "P",
    WHITE_KNIGHT: "N",
    WHITE_BISHOP: "B",
    WHITE_ROOK: "R",
    WHITE_QUEEN: "Q",
    WHITE_KING: "K",
    BLACK_PAWN: "p",
    BLACK_KNIGHT: "n",
    BLACK_BISHOP: "b",
    BLACK_ROOK: "r",
    BLACK_QUEEN: "q",
    BLACK_KING: "k",
}

PIECE_TO_ASSET = {
    WHITE_PAWN: "pawn-w.svg",
    WHITE_KNIGHT: "knight-w.svg",
    WHITE_BISHOP: "bishop-w.svg",
    WHITE_ROOK: "rook-w.svg",
    WHITE_QUEEN: "queen-w.svg",
    WHITE_KING: "king-w.svg",
    BLACK_PAWN: "pawn-b.svg",
    BLACK_KNIGHT: "knight-b.svg",
    BLACK_BISHOP: "bishop-b.svg",
    BLACK_ROOK: "rook-b.svg",
    BLACK_QUEEN: "queen-b.svg",
    BLACK_KING: "king-b.svg",
}

_PIECE_IMAGE_CACHE: dict[tuple[int, int], "Image.Image"] = {}


def configure_tf_memory_growth() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if len(gpus) > 0:
            print(f"Enabled TensorFlow memory growth on {len(gpus)} GPU(s)")
    except Exception as e:
        print(f"Could not enable TensorFlow memory growth: {e}")


def transformer_block(x, heads=4, ff_dim=128, dropout=0.1):
    attn_out = MultiHeadAttention(num_heads=heads, key_dim=MODEL_DIM // heads)(x, x)
    attn_out = Dropout(dropout)(attn_out)
    x = LayerNormalization(epsilon=1e-6)(x + attn_out)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(MODEL_DIM)(ff)
    ff = Dropout(dropout)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def residual_block_1d(x, channels: int, kernel_size: int = 3, dropout: float = 0.1):
    shortcut = x

    y = Conv1D(channels, kernel_size=kernel_size, padding="same", use_bias=False)(x)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout)(y)

    y = Conv1D(channels, kernel_size=kernel_size, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)

    x = Add()([shortcut, y])
    x = Activation("relu")(x)
    return x


def build_residual_tower_from_tokens(inp):
    tok_emb = Embedding(input_dim=NUM_TOKENS, output_dim=MODEL_DIM)(inp)
    pos_ids = tf.expand_dims(tf.range(start=0, limit=SEQ_LEN, delta=1), axis=0)
    pos_emb = Embedding(input_dim=SEQ_LEN, output_dim=MODEL_DIM)(pos_ids)
    x = tok_emb + pos_emb

    x = Conv1D(RES_TOWER_CHANNELS, kernel_size=1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    for _ in range(RES_TOWER_BLOCKS):
        x = residual_block_1d(x, channels=RES_TOWER_CHANNELS, kernel_size=3, dropout=DROPOUT_RATE)

    return x


def build_eval_model() -> keras.Model:
    inp = Input(shape=(SEQ_LEN,), dtype="int32", name="board_tokens")

    # AlphaZero-style residual trunk + value head.
    x = build_residual_tower_from_tokens(inp)

    x_avg = GlobalAveragePooling1D()(x)
    x_max = GlobalMaxPooling1D()(x)
    x = Concatenate()([x_avg, x_max])
    x = Dense(128, activation="relu")(x)
    x = Dropout(DROPOUT_RATE + 0.05)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(1, activation="tanh", name="eval_out")(x)
    model = keras.Model(inp, out, name="eval_model")
    model.compile(
        optimizer=keras.optimizers.Adam(EVAL_INITIAL_LR),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_next_board_model_transformer() -> keras.Model:
    inp = Input(shape=(SEQ_LEN,), dtype="int32", name="board_tokens")

    tok_emb = Embedding(input_dim=NUM_TOKENS, output_dim=MODEL_DIM)(inp)
    pos_ids = tf.expand_dims(tf.range(start=0, limit=SEQ_LEN, delta=1), axis=0)
    pos_emb = Embedding(input_dim=SEQ_LEN, output_dim=MODEL_DIM)(pos_ids)
    x = tok_emb + pos_emb

    for _ in range(NUM_TRANSFORMER_BLOCKS):
        x = transformer_block(x, heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT_RATE)

    x = LayerNormalization(epsilon=1e-6)(x)

    out = Dense(NUM_TOKENS, name="next_board_logits")(x)
    model = keras.Model(inp, out, name="next_board_model")
    model.compile(
        optimizer=keras.optimizers.Adam(NEXT_INITIAL_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="token_acc")],
    )
    return model


def build_next_board_model_cnn() -> keras.Model:
    inp = Input(shape=(SEQ_LEN,), dtype="int32", name="board_tokens")

    # AlphaZero-style residual trunk + policy-like head over board tokens.
    x = build_residual_tower_from_tokens(inp)
    x = Conv1D(64, kernel_size=1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    out = Dense(NUM_TOKENS, name="next_board_logits")(x)
    model = keras.Model(inp, out, name="next_board_model_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(NEXT_INITIAL_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="token_acc")],
    )
    return model


def build_next_board_model() -> keras.Model:
    if NEXT_BOARD_ARCH == "cnn":
        return build_next_board_model_cnn()
    return build_next_board_model_transformer()


def _token_index_to_square(idx: int) -> "int":
    # token idx 1..64 corresponds to board squares a8..h1
    r = (idx - 1) // 8
    c = (idx - 1) % 8
    return chess.square(c, 7 - r)


def _tokens_to_chess_board(board_tokens: np.ndarray) -> "chess.Board | None":
    if not CHESS_AVAILABLE or board_tokens.shape != (65,):
        return None

    try:
        board = chess.Board(None)
        board.clear_board()
        board.turn = chess.WHITE if int(board_tokens[0]) == WHITE_TO_MOVE else chess.BLACK
        board.castling_rights = chess.BB_EMPTY
        board.ep_square = None
        board.halfmove_clock = 0
        board.fullmove_number = 1

        for i in range(1, 65):
            token = int(board_tokens[i])
            if token == EMPTY_SQUARE:
                continue
            piece = TOKEN_TO_CHESS_PIECE.get(token)
            if piece is None:
                return None
            board.set_piece_at(_token_index_to_square(i), piece)

        return board
    except Exception:
        return None


def _chess_board_to_tokens(board: "chess.Board") -> np.ndarray:
    tokens = np.zeros((65,), dtype=np.int32)
    tokens[0] = WHITE_TO_MOVE if board.turn == chess.WHITE else BLACK_TO_MOVE
    for i in range(1, 65):
        piece = board.piece_at(_token_index_to_square(i))
        tokens[i] = CHESS_PIECE_TO_TOKEN.get(piece.symbol(), EMPTY_SQUARE) if piece else EMPTY_SQUARE
    return tokens


def _score_candidate_tokens_from_logits(logits: np.ndarray, candidate_tokens: np.ndarray) -> float:
    selected = candidate_tokens.astype(np.int64)
    stable_logits = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = stable_logits - np.log(np.sum(np.exp(stable_logits), axis=-1, keepdims=True) + 1e-12)
    return float(np.sum(log_probs[np.arange(SEQ_LEN), selected]))


def _build_legal_token_mask_for_board(board_tokens: np.ndarray) -> np.ndarray:
    """
    Build a per-position legal-token mask from python-chess legal moves.
    mask[pos, token] = 1 if token appears at `pos` in at least one legal next board.
    """
    mask = np.zeros((SEQ_LEN, NUM_TOKENS), dtype=np.float32)

    if not CHESS_AVAILABLE:
        # Fallback: allow all piece/empty tokens on squares + both turn tokens.
        mask[0, WHITE_TO_MOVE] = 1.0
        mask[0, BLACK_TO_MOVE] = 1.0
        mask[1:, : BLACK_KING + 1] = 1.0
        return mask

    board = _tokens_to_chess_board(board_tokens)
    if board is None:
        mask[0, WHITE_TO_MOVE] = 1.0
        mask[0, BLACK_TO_MOVE] = 1.0
        mask[1:, : BLACK_KING + 1] = 1.0
        return mask

    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        # Terminal board: only current tokens are possible.
        mask[np.arange(SEQ_LEN), board_tokens.astype(np.int64)] = 1.0
        return mask

    for mv in legal_moves:
        b = board.copy(stack=False)
        b.push(mv)
        nxt = _chess_board_to_tokens(b)
        mask[np.arange(SEQ_LEN), nxt.astype(np.int64)] = 1.0

    return mask


def _build_legal_token_mask_batch(x_batch: np.ndarray) -> np.ndarray:
    masks = np.zeros((x_batch.shape[0], SEQ_LEN, NUM_TOKENS), dtype=np.float32)
    for i in range(x_batch.shape[0]):
        masks[i] = _build_legal_token_mask_for_board(x_batch[i])
    return masks


def train_next_board_step(
    next_board_model: keras.Model,
    x_in: np.ndarray,
    y_board_batch: np.ndarray,
) -> dict:
    if not CHESS_AVAILABLE:
        return next_board_model.train_on_batch(x_in, y_board_batch, return_dict=True)

    legal_mask_np = _build_legal_token_mask_batch(x_in)
    legal_mask = tf.convert_to_tensor(legal_mask_np, dtype=tf.float32)

    with tf.GradientTape() as tape:
        logits = next_board_model(x_in, training=True)

        ce = tf.keras.losses.sparse_categorical_crossentropy(y_board_batch, logits, from_logits=True)
        ce_loss = tf.reduce_mean(ce)

        probs = tf.nn.softmax(logits, axis=-1)
        legal_mass = tf.reduce_sum(probs * legal_mask, axis=-1)
        legal_mass_loss = -tf.reduce_mean(tf.math.log(legal_mass + 1e-8))

        loss = ce_loss + LEGAL_LOSS_WEIGHT * legal_mass_loss

    grads = tape.gradient(loss, next_board_model.trainable_variables)
    next_board_model.optimizer.apply_gradients(zip(grads, next_board_model.trainable_variables))

    token_acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_board_batch, logits))

    return {
        "loss": float(loss.numpy()),
        "token_acc": float(token_acc.numpy()),
        "ce_loss": float(ce_loss.numpy()),
        "legal_mass_loss": float(legal_mass_loss.numpy()),
    }


def _predict_next_board_fallback(logits: np.ndarray, board_tokens: np.ndarray) -> np.ndarray:
    next_tokens = np.zeros((SEQ_LEN,), dtype=np.int32)
    next_tokens[0] = BLACK_TO_MOVE if int(board_tokens[0]) == WHITE_TO_MOVE else WHITE_TO_MOVE

    for i in range(1, SEQ_LEN):
        square_logits = logits[i, : BLACK_KING + 1].astype(np.float64)
        temp = max(1e-6, SELF_PLAY_TEMPERATURE)
        square_logits = square_logits / temp

        if SELF_PLAY_TOP_K > 0 and SELF_PLAY_TOP_K < square_logits.shape[0]:
            topk_idx = np.argpartition(square_logits, -SELF_PLAY_TOP_K)[-SELF_PLAY_TOP_K:]
            filtered = np.full_like(square_logits, -1e9)
            filtered[topk_idx] = square_logits[topk_idx]
            square_logits = filtered

        square_logits = square_logits - np.max(square_logits)
        probs = np.exp(square_logits)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            next_tokens[i] = int(np.argmax(logits[i, : BLACK_KING + 1]))
        else:
            probs = probs / probs_sum
            next_tokens[i] = int(np.random.choice(np.arange(BLACK_KING + 1), p=probs))

    next_tokens[1:] = np.clip(next_tokens[1:], 0, BLACK_KING)
    return next_tokens


def _predict_next_board_from_eval(eval_model: keras.Model, board_tokens: np.ndarray) -> np.ndarray:
    if not CHESS_AVAILABLE:
        return board_tokens.copy()

    board = _tokens_to_chess_board(board_tokens)
    if board is None:
        return board_tokens.copy()

    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return board_tokens.copy()

    candidates: list[np.ndarray] = []
    for mv in legal_moves:
        b = board.copy(stack=False)
        b.push(mv)
        candidates.append(_chess_board_to_tokens(b))

    stacked = np.stack(candidates, axis=0)

    if SELF_PLAY_USE_DROPOUT_INFERENCE:
        sample_evals = []
        for _ in range(max(1, SELF_PLAY_MC_DROPOUT_SAMPLES)):
            sample = eval_model(stacked, training=True).numpy().reshape(-1)
            sample_evals.append(sample)
        evals = np.mean(np.stack(sample_evals, axis=0), axis=0)
    else:
        evals = eval_model.predict(stacked, verbose=0).reshape(-1)

    # Eval is normalized white perspective. White to move maximizes, black to move minimizes.
    if int(board_tokens[0]) == WHITE_TO_MOVE:
        idx = int(np.argmax(evals))
    else:
        idx = int(np.argmin(evals))

    return candidates[idx]


def fetch_batch(batch_size: int):
    x_batch, y_board_batch, y_eval_batch = [], [], []
    failures = 0

    while len(x_batch) < batch_size:
        try:
            resp = requests.get(API_URL, timeout=10)
            resp.raise_for_status()
            parsed = parse_board_response(resp.json())
            x_tokens = parsed["X"].astype(np.int32)
            x_batch.append(x_tokens)
            y_board_batch.append(parsed["Y"].astype(np.int32))
            eval_value = float(parsed["eval"])

            # Stockfish score is side-to-move perspective. Convert to white perspective if configured.
            if EVAL_PERSPECTIVE == "white" and int(x_tokens[0]) == BLACK_TO_MOVE:
                eval_value = -eval_value

            # Stable chess eval normalization: y = tanh(cp / k), here eval_value is in pawns.
            eval_value = float(np.tanh(eval_value / EVAL_TANH_K_PAWNS))

            y_eval_batch.append(np.float32(eval_value))
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


def compute_error_rate_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = ERROR_RATE_EPS) -> float:
    y_true = y_true.astype(np.float32).reshape(-1)
    y_pred = y_pred.astype(np.float32).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    pct = np.abs(y_pred - y_true) / denom
    return float(np.mean(pct) * 100.0)


def compute_filtered_error_rate_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_abs_target: float = ERROR_RATE_MIN_ABS_TARGET,
    eps: float = ERROR_RATE_EPS,
) -> float:
    y_true = y_true.astype(np.float32).reshape(-1)
    y_pred = y_pred.astype(np.float32).reshape(-1)
    mask = np.abs(y_true) >= float(min_abs_target)
    if not np.any(mask):
        return 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yp - yt) / denom) * 100.0)


def compute_smape_pct(y_true: np.ndarray, y_pred: np.ndarray, eps: float = ERROR_RATE_EPS) -> float:
    y_true = y_true.astype(np.float32).reshape(-1)
    y_pred = y_pred.astype(np.float32).reshape(-1)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    smape = 200.0 * np.abs(y_pred - y_true) / denom
    return float(np.mean(smape))


def denormalize_eval(y_norm: np.ndarray) -> np.ndarray:
    # Inverse of y = tanh(cp / k), returns eval in pawns.
    y_norm = np.clip(y_norm.astype(np.float32), -0.999999, 0.999999)
    return np.arctanh(y_norm) * EVAL_TANH_K_PAWNS


def normalize_eval(y_raw: np.ndarray) -> np.ndarray:
    y_raw = y_raw.astype(np.float32)
    return np.tanh(y_raw / EVAL_TANH_K_PAWNS)


def load_or_create_validation_batch(cache_path: str, batch_size: int):
    try:
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            x_val = data["x"]
            y_board_val = data["y_board"]
            y_eval_val = data["y_eval"]
            print(f"Loaded validation batch from {cache_path} (n={x_val.shape[0]})")
            return x_val, y_board_val, y_eval_val
    except Exception as e:
        print(f"Failed loading validation cache {cache_path}: {e}")

    x_val, y_board_val, y_eval_val = fetch_batch(batch_size)
    if x_val is None:
        print("Validation batch not available from API")
        return None, None, None

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, x=x_val, y_board=y_board_val, y_eval=y_eval_val)
    print(f"Created validation batch cache at {cache_path} (n={x_val.shape[0]})")
    return x_val, y_board_val, y_eval_val


def _get_optimizer_lr(optimizer: keras.optimizers.Optimizer) -> float:
    lr = optimizer.learning_rate
    try:
        return float(tf.keras.backend.get_value(lr))
    except Exception:
        return float(lr)


def _set_optimizer_lr(optimizer: keras.optimizers.Optimizer, new_lr: float) -> None:
    lr = optimizer.learning_rate
    try:
        tf.keras.backend.set_value(lr, new_lr)
    except Exception:
        optimizer.learning_rate = new_lr


def _try_resume_from_best_checkpoint(model: keras.Model, checkpoint_path: str, model_name: str) -> bool:
    if not RESUME_FROM_BEST_CHECKPOINT:
        return False
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found for {model_name} at {checkpoint_path}")
        return False

    try:
        loaded = keras.models.load_model(checkpoint_path, compile=False)
        model.set_weights(loaded.get_weights())
        print(f"Resumed {model_name} weights from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Failed to resume {model_name} from {checkpoint_path}: {e}")
        return False


def _resume_if_enabled(model: keras.Model, checkpoint_path: str, model_name: str, enabled: bool) -> bool:
    if not enabled:
        print(f"Resume disabled for {model_name}; training from scratch")
        return False
    return _try_resume_from_best_checkpoint(model, checkpoint_path, model_name)


def _prepare_model_dirs_and_migrate_legacy() -> None:
    os.makedirs(os.path.dirname(BEST_EVAL_CKPT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BEST_NEXT_CKPT_PATH), exist_ok=True)

    if os.path.exists(LEGACY_BEST_EVAL_CKPT_PATH) and not os.path.exists(BEST_EVAL_CKPT_PATH):
        shutil.move(LEGACY_BEST_EVAL_CKPT_PATH, BEST_EVAL_CKPT_PATH)
        print(f"Moved legacy eval checkpoint -> {BEST_EVAL_CKPT_PATH}")

    if os.path.exists(LEGACY_BEST_NEXT_CKPT_PATH) and not os.path.exists(BEST_NEXT_CKPT_PATH):
        shutil.move(LEGACY_BEST_NEXT_CKPT_PATH, BEST_NEXT_CKPT_PATH)
        print(f"Moved legacy next-board checkpoint -> {BEST_NEXT_CKPT_PATH}")


def _default_start_board_tokens() -> np.ndarray:
    board = np.zeros((65,), dtype=np.int32)
    board[0] = WHITE_TO_MOVE
    board[1:9] = np.array(
        [BLACK_ROOK, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING, BLACK_BISHOP, BLACK_KNIGHT, BLACK_ROOK],
        dtype=np.int32,
    )
    board[9:17] = BLACK_PAWN
    board[49:57] = WHITE_PAWN
    board[57:65] = np.array(
        [WHITE_ROOK, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING, WHITE_BISHOP, WHITE_KNIGHT, WHITE_ROOK],
        dtype=np.int32,
    )
    return board


def _predict_next_board(next_board_model: keras.Model, board_tokens: np.ndarray) -> np.ndarray:
    # Monte Carlo dropout + legal-move constrained decoding via python-chess
    logits = next_board_model(board_tokens[None, :], training=True).numpy()[0]
    if CHESS_AVAILABLE:
        board = _tokens_to_chess_board(board_tokens)
        if board is not None:
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 0:
                return board_tokens.copy()

            candidates: list[np.ndarray] = []
            scores: list[float] = []
            for mv in legal_moves:
                b = board.copy(stack=False)
                b.push(mv)
                candidate = _chess_board_to_tokens(b)
                candidates.append(candidate)
                scores.append(_score_candidate_tokens_from_logits(logits, candidate))

            scores_np = np.array(scores, dtype=np.float64)
            temp = max(1e-6, SELF_PLAY_TEMPERATURE)
            scores_np = scores_np / temp

            if SELF_PLAY_TOP_K > 0 and SELF_PLAY_TOP_K < scores_np.shape[0]:
                topk_idx = np.argpartition(scores_np, -SELF_PLAY_TOP_K)[-SELF_PLAY_TOP_K:]
                filtered = np.full_like(scores_np, -1e9)
                filtered[topk_idx] = scores_np[topk_idx]
                scores_np = filtered

            scores_np = scores_np - np.max(scores_np)
            probs = np.exp(scores_np)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                choice = int(np.argmax(scores_np))
            else:
                probs = probs / probs_sum
                choice = int(np.random.choice(np.arange(len(candidates)), p=probs))

            return candidates[choice]

    return _predict_next_board_fallback(logits, board_tokens)


def _render_board_frame(board_tokens: np.ndarray, ply: int, eval_score: float) -> "Image.Image":
    cell = 64
    header_h = 56
    board_px = cell * 8
    img = Image.new("RGB", (board_px, board_px + header_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    light = (240, 217, 181)
    dark = (181, 136, 99)
    white_piece = (245, 245, 245)
    black_piece = (35, 35, 35)

    side_txt = "white" if int(board_tokens[0]) == WHITE_TO_MOVE else "black"
    draw.text((8, 8), f"Self-play | ply {ply} | to move: {side_txt} | eval: {eval_score:.3f}", fill=(255, 255, 255), font=font)

    squares = board_tokens[1:65].reshape(8, 8)

    def _load_piece_image(token: int, target_px: int) -> "Image.Image | None":
        if not PIL_AVAILABLE:
            return None
        key = (token, target_px)
        if key in _PIECE_IMAGE_CACHE:
            return _PIECE_IMAGE_CACHE[key]

        asset_name = PIECE_TO_ASSET.get(token)
        if not asset_name:
            return None

        asset_path = os.path.join("assets", asset_name)
        if not os.path.exists(asset_path):
            return None

        try:
            if not CAIROSVG_AVAILABLE:
                return None

            png_bytes = cairosvg.svg2png(url=asset_path)
            img = Image.open(BytesIO(png_bytes)).convert("RGBA")

            # Fit into cell with margin while preserving aspect ratio.
            max_side = max(1, int(target_px * 0.82))
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            _PIECE_IMAGE_CACHE[key] = img
            return img
        except Exception:
            return None

    for r in range(8):
        for c in range(8):
            x0 = c * cell
            y0 = header_h + r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=light if (r + c) % 2 == 0 else dark)

            token = int(squares[r, c])
            piece_img = _load_piece_image(token, cell)
            if piece_img is not None:
                px = x0 + (cell - piece_img.width) // 2
                py = y0 + (cell - piece_img.height) // 2
                img.paste(piece_img, (px, py), piece_img)
            else:
                glyph = PIECE_TO_GLYPH.get(token, "?")
                if glyph:
                    fill = white_piece if token <= WHITE_KING else black_piece
                    draw.text((x0 + cell // 2 - 4, y0 + cell // 2 - 6), glyph, fill=fill, font=font)

    return img


def create_and_log_self_play_gif(
    eval_model: keras.Model,
    step: int,
    seed_board: np.ndarray | None = None,
) -> None:
    try:
        if not PIL_AVAILABLE:
            wandb.log({"self_play/gif_skipped": 1}, step=step)
            return

        os.makedirs("artifacts/self_play", exist_ok=True)

        if SELF_PLAY_ALWAYS_FROM_START:
            current = _default_start_board_tokens()
        else:
            current = (
                seed_board.astype(np.int32).copy()
                if seed_board is not None and seed_board.shape == (65,)
                else _default_start_board_tokens()
            )

        frames = []
        evals = []
        for ply in range(SELF_PLAY_PLIES + 1):
            ev = float(eval_model(current[None, :], training=SELF_PLAY_USE_DROPOUT_INFERENCE).numpy()[0][0])
            evals.append(ev)
            frames.append(_render_board_frame(current, ply=ply, eval_score=ev))
            if ply < SELF_PLAY_PLIES:
                if CHESS_AVAILABLE:
                    board = _tokens_to_chess_board(current)
                    if board is not None and board.is_game_over():
                        break
                current = _predict_next_board_from_eval(eval_model, current)

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        gif_path = f"artifacts/self_play/self_play_step_{step}_{ts}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=800,
            loop=0,
            optimize=False,
        )

        wandb.log(
            {
                "self_play/gif": wandb.Video(gif_path, format="gif"),
                "self_play/start_eval": float(evals[0]),
                "self_play/end_eval": float(evals[-1]),
            },
            step=step,
        )
    except Exception as e:
        print(f"Step {step}: failed to create/log self-play GIF: {e}")
        wandb.log({"self_play/gif_skipped": 1}, step=step)


if __name__ == "__main__":
    configure_tf_memory_growth()
    _prepare_model_dirs_and_migrate_legacy()

    wandb.init(
        project="chess-transformer-eval-only",
        config={
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "eval_lr": EVAL_INITIAL_LR,
            "eval_perspective": EVAL_PERSPECTIVE,
            "eval_norm": "tanh",
            "eval_tanh_k_pawns": EVAL_TANH_K_PAWNS,
            "validate_every_steps": VALIDATE_EVERY_STEPS,
            "val_batch_size": VAL_BATCH_SIZE,
            "val_cache_path": VAL_CACHE_PATH,
            "res_tower_channels": RES_TOWER_CHANNELS,
            "res_tower_blocks": RES_TOWER_BLOCKS,
            "best_eval_ckpt_path": BEST_EVAL_CKPT_PATH,
            "model_dim": MODEL_DIM,
            "num_heads": NUM_HEADS,
            "ff_dim": FF_DIM,
            "num_transformer_blocks": NUM_TRANSFORMER_BLOCKS,
            "dropout_rate": DROPOUT_RATE,
            "self_play_plies": SELF_PLAY_PLIES,
            "lr_reduce_patience_steps": LR_REDUCE_PATIENCE_STEPS,
            "lr_reduce_factor": LR_REDUCE_FACTOR,
            "train_mode": "eval_only",
        },
    )

    eval_model = build_eval_model()

    eval_resumed = _resume_if_enabled(
        model=eval_model,
        checkpoint_path=BEST_EVAL_CKPT_PATH,
        model_name="eval_model",
        enabled=RESUME_EVAL_MODEL,
    )

    wandb.run.summary["resumed_eval_model"] = bool(eval_resumed)

    eval_model.summary()

    # Log model summaries to W&B
    eval_summary_text = get_model_summary_text(eval_model)

    with open("eval_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(eval_summary_text)

    wandb.save("eval_model_summary.txt")
    wandb.run.summary["eval_model_params"] = int(eval_model.count_params())
    wandb.run.summary["target_eval_model_params"] = 5_000_000

    x_val, _y_board_val, y_eval_val = load_or_create_validation_batch(VAL_CACHE_PATH, VAL_BATCH_SIZE)
    has_val = x_val is not None
    wandb.run.summary["has_validation_batch"] = bool(has_val)

    best_eval_loss = float("inf")
    last_eval_improvement_step = 0

    for step in range(STEPS):
        x_batch, y_board_batch, y_eval_batch = fetch_batch(BATCH_SIZE)
        if x_batch is None:
            print(f"Step {step + 1}/{STEPS}: no batch fetched")
            continue

        x_in = x_batch

        changed_counts = np.sum(x_batch[:, 1:] != y_board_batch[:, 1:], axis=1)
        avg_changed_squares = float(np.mean(changed_counts))
        min_changed_squares = int(np.min(changed_counts))
        max_changed_squares = int(np.max(changed_counts))
        eval_target_mean = float(np.mean(y_eval_batch))
        eval_target_min = float(np.min(y_eval_batch))
        eval_target_max = float(np.max(y_eval_batch))

        eval_metrics = eval_model.train_on_batch(x_in, y_eval_batch, return_dict=True)
        eval_pred_batch = eval_model.predict(x_in, verbose=0).reshape(-1)
        eval_mae_batch = float(np.mean(np.abs(eval_pred_batch - y_eval_batch)))
        eval_loss_batch = float(np.mean(np.square(eval_pred_batch - y_eval_batch)))
        eval_error_rate_pct = compute_error_rate_pct(y_eval_batch, eval_pred_batch)
        eval_error_rate_filtered_pct = compute_filtered_error_rate_pct(y_eval_batch, eval_pred_batch)
        eval_smape_pct = compute_smape_pct(y_eval_batch, eval_pred_batch)
        eval_mae_raw = float(np.mean(np.abs(denormalize_eval(eval_pred_batch) - denormalize_eval(y_eval_batch))))

        wandb.log(
            {
                "step": step + 1,
                "eval/loss": eval_loss_batch,
                "eval/mae": eval_mae_batch,
                "eval/loss_running": float(eval_metrics["loss"]),
                "eval/mae_running": float(eval_metrics["mae"]),
                "eval/error_rate_pct": eval_error_rate_pct,
                "eval/error_rate_filtered_pct": eval_error_rate_filtered_pct,
                "eval/smape_pct": eval_smape_pct,
                "eval/mae_raw": eval_mae_raw,
                "data/eval_target_mean": eval_target_mean,
                "data/eval_target_min": eval_target_min,
                "data/eval_target_max": eval_target_max,
                "data/avg_changed_squares": avg_changed_squares,
                "data/min_changed_squares": min_changed_squares,
                "data/max_changed_squares": max_changed_squares,
                "batch_size": int(x_batch.shape[0]),
            },
            step=step + 1,
        )

        if has_val and ((step + 1) % VALIDATE_EVERY_STEPS == 0):
            val_pred = eval_model.predict(x_val, verbose=0).reshape(-1)
            val_mae = float(np.mean(np.abs(val_pred - y_eval_val)))
            val_loss = float(np.mean(np.square(val_pred - y_eval_val)))
            val_smape = compute_smape_pct(y_eval_val, val_pred)
            val_mae_raw = float(np.mean(np.abs(denormalize_eval(val_pred) - denormalize_eval(y_eval_val))))

            wandb.log(
                {
                    "val/eval_loss": val_loss,
                    "val/eval_mae": val_mae,
                    "val/eval_mae_raw": val_mae_raw,
                    "val/eval_smape_pct": val_smape,
                },
                step=step + 1,
            )

        if (step + 1) % SELF_PLAY_GIF_EVERY_STEPS == 0:
            seed_board = x_batch[0] if x_batch.shape[0] > 0 else None
            create_and_log_self_play_gif(
                eval_model=eval_model,
                step=step + 1,
                seed_board=seed_board,
            )

        # Save best checkpoints
        eval_improved = False
        if eval_loss_batch < best_eval_loss:
            best_eval_loss = eval_loss_batch
            eval_model.save(BEST_EVAL_CKPT_PATH)
            wandb.run.summary["best_eval_loss"] = best_eval_loss
            wandb.run.summary["best_eval_step"] = step + 1
            eval_improved = True

        if eval_improved:
            last_eval_improvement_step = step + 1

        reduced_eval = 0

        if (step + 1) - last_eval_improvement_step >= LR_REDUCE_PATIENCE_STEPS:
            eval_lr = _get_optimizer_lr(eval_model.optimizer)
            new_eval_lr = eval_lr * LR_REDUCE_FACTOR
            _set_optimizer_lr(eval_model.optimizer, new_eval_lr)
            last_eval_improvement_step = step + 1
            reduced_eval = 1
            print(
                f"Step {step + 1}: eval plateau {LR_REDUCE_PATIENCE_STEPS} steps -> eval_lr={new_eval_lr:.8f}"
            )

        wandb.log(
            {
                "lr/eval": _get_optimizer_lr(eval_model.optimizer),
                "lr/eval_reduced_on_plateau": reduced_eval,
            },
            step=step + 1,
        )

        print(
            f"Step {step + 1}/{STEPS} | "
            f"eval_loss={eval_loss_batch:.4f} "
            f"eval_mae={eval_mae_batch:.4f}"
        )

    wandb.finish()
