import chess
import chess.pgn
import requests
import os
import json
import time

BACKEND_URL = "http://localhost:1323/postBoard"
ELO_THRESHOLD = 1950
CHECKPOINT_INTERVAL = 1
DELAY_BETWEEN_REQUESTS = 0.05  # 20 requests/sec
session = requests.Session()

def processGames(licheesDatasetPath: str, start_idx: int = 0):
    if not os.path.isfile(licheesDatasetPath):
        raise FileNotFoundError(f"PGN file not found: {licheesDatasetPath}")

    total_sent = 0
    current_idx = 0
    start_time = time.time()
    last_checkpoint_time = start_time

    with open(licheesDatasetPath, "r", encoding="utf-8", errors="ignore") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            headers = game.headers
            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                continue

            if white_elo < ELO_THRESHOLD or black_elo < ELO_THRESHOLD:
                continue

            board = game.board()
            game_id = headers.get("Site") or headers.get("LichessID") or ""

            for move in game.mainline_moves():
                # Skip already processed
                if current_idx < start_idx:
                    current_idx += 1
                    board.push(move)
                    continue

                before_fen = board.fen()
                mover = "white" if board.turn == chess.WHITE else "black"
                board.push(move)
                after_fen = board.fen()

                payload = {
                    "X": before_fen,
                    "Y": after_fen,
                    "player": mover,
                    "move": move.uci(),
                    "game_id": game_id,
                }

                if postBoard(payload):
                    total_sent += 1
                
                # Rate limit to avoid overwhelming server
                time.sleep(DELAY_BETWEEN_REQUESTS)
                
                current_idx += 1

                # Save checkpoint periodically
                if current_idx % CHECKPOINT_INTERVAL == 0:
                    current_time = time.time()
                    elapsed = current_time - last_checkpoint_time
                    rate = CHECKPOINT_INTERVAL / elapsed if elapsed > 0 else 0
                    total_elapsed = current_time - start_time
                    
                    print(f"Progress: {current_idx} processed, {total_sent} sent | Rate: {rate:.1f} moves/sec | Elapsed: {total_elapsed:.1f}s")
                    
                    with open('check_point.json', 'w') as f:
                        json.dump({"last_idx": current_idx}, f)
                    
                    last_checkpoint_time = current_time

    return current_idx, total_sent

def postBoard(payload: dict) -> bool:
    try:
        response = session.post(BACKEND_URL, json=payload, timeout=15)
        response.raise_for_status()
        return True
    except requests.Timeout:
        print(f"Timeout sending move {payload['move']}")
        return False
    except requests.RequestException as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    licheesDsPath = "/media/user/2TB Clear/chessData/lichess_db_standard_rated_2026-01.pgn"
    last_idx = 0

    # Load checkpoint if exists
    if os.path.exists('check_point.json'):
        try:
            with open('check_point.json', 'r') as f:
                checkpoint = json.load(f)
                if checkpoint.get("last_idx") is not None:
                    last_idx = checkpoint["last_idx"]
        except (json.JSONDecodeError, ValueError):
            last_idx = 0

    print(f"Starting from index {last_idx}")
    
    current_idx, total_sent = processGames(licheesDsPath, start_idx=last_idx)
    
    # Final checkpoint
    with open('check_point.json', 'w') as f:
        json.dump({"last_idx": current_idx}, f)
    
    print(f"Completed! Processed {current_idx} moves, sent {total_sent} to API")