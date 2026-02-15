from train import fetch_batch
from pprint import pprint
from dataHelper import tokens_to_board, board_to_feature_planes
import chess
import random
import cachetools
from tensorflow import keras
import time
import numpy as np

cache = cachetools.LRUCache(maxsize=100_000)
search_cache = cachetools.LRUCache(maxsize=500_000)
model = None
DEBUG_EVAL = False
NEG_INF = float("-inf")
POS_INF = float("inf")


def _model_predict_batch(token_batch: np.ndarray) -> np.ndarray:
    global model
    if model is None:
        return np.random.randint(-10, 11, size=(token_batch.shape[0],)).astype(np.float32)

    preds = model(token_batch.astype(np.float32), training=False).numpy().reshape(-1)
    return preds.astype(np.float32)

def evaluate(board: chess.Board) -> float:
    if board.is_checkmate():
        return -1000
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0

    plane_batch = board_to_feature_planes(board).reshape(1, 8, 8, 54).astype(np.float32)
    output = float(_model_predict_batch(plane_batch)[0])
    if DEBUG_EVAL:
        print(f"Evaluating board: {board.fen()} -> {output}")
    return output

def evalBoard(board: chess.Board) -> float:
    board_fen = board.fen()
    if board_fen in cache:
        return cache[board_fen]
    
    score = evaluate(board)
    cache[board_fen] = score
    return score


def evalBoardsBatch(boards: list[chess.Board]) -> list[float]:
    if len(boards) == 0:
        return []

    out: list[float | None] = [None] * len(boards)
    missing_idx: list[int] = []
    missing_planes: list[np.ndarray] = []

    for i, b in enumerate(boards):
        fen = b.fen()
        if fen in cache:
            out[i] = float(cache[fen])
            continue

        if b.is_checkmate():
            score = -1000.0
            cache[fen] = score
            out[i] = score
            continue
        if b.is_stalemate() or b.is_insufficient_material():
            score = 0.0
            cache[fen] = score
            out[i] = score
            continue

        missing_idx.append(i)
        missing_planes.append(board_to_feature_planes(b).astype(np.float32))

    if len(missing_planes) > 0:
        batch = np.stack(missing_planes, axis=0)
        preds = _model_predict_batch(batch)
        for j, i in enumerate(missing_idx):
            score = float(preds[j])
            fen = boards[i].fen()
            cache[fen] = score
            out[i] = score

    return [float(v) for v in out]


def _select_top_k_moves(board: chess.Board, legal_moves: list[chess.Move], top_k: int | None) -> list[chess.Move]:
    if top_k is None or top_k <= 0 or top_k >= len(legal_moves):
        return legal_moves

    child_boards: list[chess.Board] = []
    for move in legal_moves:
        board.push(move)
        child_boards.append(board.copy(stack=False))
        board.pop()

    child_scores = evalBoardsBatch(child_boards)

    ranked: list[tuple[float, chess.Move]] = []
    for move, child_score in zip(legal_moves, child_scores):
        # Child node is opponent-to-move, so negate for current side ordering.
        approx_score = -child_score
        ranked.append((approx_score, move))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [mv for _, mv in ranked[:top_k]]
        
def _search_score(board: chess.Board, depth: int, alpha: float, beta: float, top_k: int | None) -> float:
    if depth == 0 or board.is_game_over():
        return evalBoard(board)

    cache_key = (board.fen(), depth)
    cached = search_cache.get(cache_key)
    if cached is not None:
        return float(cached)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return evalBoard(board)

    candidate_moves = _select_top_k_moves(board, legal_moves, top_k)
    best_score = NEG_INF
    for move in candidate_moves:
        board.push(move)
        score = -_search_score(board, depth - 1, -beta, -alpha, top_k)
        board.pop()

        if score > best_score:
            best_score = score
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            break

    search_cache[cache_key] = best_score
    return best_score


def search(board: chess.Board, depth: int, return_move: bool = True, top_k: int | None = None):
    if depth == 0 or board.is_game_over():
        return evalBoard(board)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None if return_move else evalBoard(board)

    candidate_moves = _select_top_k_moves(board, legal_moves, top_k)
    alpha = NEG_INF
    beta = POS_INF
    best_move = candidate_moves[0]
    best_score = NEG_INF

    for move in candidate_moves:
        board.push(move)
        score = -_search_score(board, depth - 1, -beta, -alpha, top_k)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    return best_move if return_move else best_score


def bestBoard(board: chess.Board, depth: int, top_k: int | None = None) -> tuple[chess.Board, float, float]:
    start = time.time()
    best_move = search(board, depth, top_k=top_k)
    if best_move is None:
        return board, float(evalBoard(board)), time.time() - start  # No legal moves
    board.push(best_move)
    best_score = -float(_search_score(board, depth - 1, NEG_INF, POS_INF, top_k))
    board.pop()

    board.push(best_move)
    return board, best_score, time.time() - start

def loadModel(path: str):
    global model
    model = keras.models.load_model(path)

if __name__ == "__main__":
    x_batch, y_batch, y_eval_batch = fetch_batch(1)  # Fetch a single sample batch
    if x_batch is None:
        raise RuntimeError("fetch_batch returned no data")

    loadModel("models/eval_cnn/best_eval_model.keras")

    x = x_batch[0]
    y = y_batch[0]
    y_eval = y_eval_batch[0]

    pprint(x)
    print("-" * 50)
    pprint(y)
    print("-" * 50)
    pprint(y_eval)
    print("-" * 50)
    board = tokens_to_board(x)
    print(board)
    print("-" * 50)
    board = tokens_to_board(y)
    print(board)
    print("-" * 50)
    best_next_board, best_score, elapsed_time = bestBoard(tokens_to_board(x), depth=8, top_k=2)
    print(best_next_board)
    print(f"best_score model: {best_score} (ground truth eval: {y_eval})")
    print(f"Elapsed time: {elapsed_time} seconds")
    best_next_board, best_score, elapsed_time = bestBoard(tokens_to_board(x), depth=8, top_k=2)
    print(best_next_board)
    print(f"best_score model: {best_score} (ground truth eval: {y_eval})")
    print(f"Elapsed time: {elapsed_time} seconds")
    
    
