import numpy as np
import chess
import base64

BOARD_SIZE = 8
FEATURE_PLANES = 54

def bytes_to_tokens(board_bytes: bytes) -> np.ndarray:
    """
    Convert tokenized board bytes to numpy array of integers.
    
    Args:
        board_bytes: 65 bytes where:
            - byte[0]: turn (13=white to move, 14=black to move)
            - bytes[1-64]: board positions
    
    Returns:
        numpy array of 65 integers
    """
    return np.frombuffer(board_bytes, dtype=np.uint8)


def tokens_to_board(tokens: np.ndarray) -> chess.Board:
    """
    Convert a numpy array of tokens back to a chess.Board object.
    
    Args:
        tokens: numpy array of 65 integers (first is turn, rest are pieces)
    
    Returns:
        chess.Board object representing the position
    """
    board = chess.Board.empty()
    
    # Set turn
    if tokens[0] == WHITE_TO_MOVE:
        board.turn = chess.WHITE
    elif tokens[0] == BLACK_TO_MOVE:
        board.turn = chess.BLACK
    else:
        raise ValueError("Invalid turn token")
    
    # Map tokens to pieces and set them on the board
    for i in range(1, 65):
        token = tokens[i]
        if token == EMPTY_SQUARE:
            continue
        
        piece_type = None
        color = None
        
        if token in (WHITE_PAWN, BLACK_PAWN):
            piece_type = chess.PAWN
        elif token in (WHITE_KNIGHT, BLACK_KNIGHT):
            piece_type = chess.KNIGHT
        elif token in (WHITE_BISHOP, BLACK_BISHOP):
            piece_type = chess.BISHOP
        elif token in (WHITE_ROOK, BLACK_ROOK):
            piece_type = chess.ROOK
        elif token in (WHITE_QUEEN, BLACK_QUEEN):
            piece_type = chess.QUEEN
        elif token in (WHITE_KING, BLACK_KING):
            piece_type = chess.KING
        
        if token <= WHITE_KING:
            color = chess.WHITE
        else:
            color = chess.BLACK
        
        square_index = i - 1  # Adjust for turn byte at index 0
        square = chess.SQUARES[square_index]
        
        board.set_piece_at(square, chess.Piece(piece_type, color))
    
    return board

def board_to_tokens(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board object to a numpy array of tokens.
    
    Args:
        board: chess.Board object to convert
    
    Returns:
        numpy array of 65 integers (first is turn, rest are pieces)
    """
    tokens = np.zeros(65, dtype=np.uint8)
    
    # Set turn token
    tokens[0] = WHITE_TO_MOVE if board.turn == chess.WHITE else BLACK_TO_MOVE
    
    # Map pieces to tokens
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        piece_type = piece.piece_type
        color = piece.color
        
        token = None
        if piece_type == chess.PAWN:
            token = WHITE_PAWN if color == chess.WHITE else BLACK_PAWN
        elif piece_type == chess.KNIGHT:
            token = WHITE_KNIGHT if color == chess.WHITE else BLACK_KNIGHT
        elif piece_type == chess.BISHOP:
            token = WHITE_BISHOP if color == chess.WHITE else BLACK_BISHOP
        elif piece_type == chess.ROOK:
            token = WHITE_ROOK if color == chess.WHITE else BLACK_ROOK
        elif piece_type == chess.QUEEN:
            token = WHITE_QUEEN if color == chess.WHITE else BLACK_QUEEN
        elif piece_type == chess.KING:
            token = WHITE_KING if color == chess.WHITE else BLACK_KING
        
        tokens[square + 1] = token  # +1 to account for turn byte at index 0
    
    return tokens


def _square_row_col(square: int) -> tuple[int, int]:
    return 7 - chess.square_rank(square), chess.square_file(square)


def _set_plane_square(planes: np.ndarray, plane_idx: int, square: int, value: float = 1.0) -> None:
    r, c = _square_row_col(square)
    planes[r, c, plane_idx] = value


def board_to_feature_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((BOARD_SIZE, BOARD_SIZE, FEATURE_PLANES), dtype=np.float32)

    piece_plane = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for square, piece in board.piece_map().items():
        _set_plane_square(planes, piece_plane[(piece.piece_type, piece.color)], square)

    planes[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[:, :, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[:, :, 14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[:, :, 15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[:, :, 16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        _set_plane_square(planes, 17, board.ep_square)

    white_attackers_count = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    black_attackers_count = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for square in chess.SQUARES:
        wr, wc = _square_row_col(square)
        white_attackers_count[wr, wc] = float(len(board.attackers(chess.WHITE, square)))
        black_attackers_count[wr, wc] = float(len(board.attackers(chess.BLACK, square)))
        if board.is_attacked_by(chess.WHITE, square):
            _set_plane_square(planes, 18, square)
        if board.is_attacked_by(chess.BLACK, square):
            _set_plane_square(planes, 19, square)

    for square in board.pieces(chess.PAWN, chess.WHITE):
        for target in board.attacks(square):
            _set_plane_square(planes, 20, target)
    for square in board.pieces(chess.PAWN, chess.BLACK):
        for target in board.attacks(square):
            _set_plane_square(planes, 21, target)

    white_board = board.copy(stack=False)
    white_board.turn = chess.WHITE
    black_board = board.copy(stack=False)
    black_board.turn = chess.BLACK
    white_legal_moves = list(white_board.legal_moves)
    black_legal_moves = list(black_board.legal_moves)
    planes[:, :, 22] = min(1.0, len(white_legal_moves) / 64.0)
    planes[:, :, 23] = min(1.0, len(black_legal_moves) / 64.0)

    piece_type_to_plane = {
        chess.PAWN: 24,
        chess.KNIGHT: 25,
        chess.BISHOP: 26,
        chess.ROOK: 27,
        chess.QUEEN: 28,
        chess.KING: 29,
    }
    for move in white_legal_moves + black_legal_moves:
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        _set_plane_square(planes, piece_type_to_plane[piece.piece_type], move.to_square, 1.0)

    king_zone_offsets = (-9, -8, -7, -1, 0, 1, 7, 8, 9)
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    if white_king_sq is not None:
        for offset in king_zone_offsets:
            sq = white_king_sq + offset
            if 0 <= sq < 64 and abs(chess.square_file(sq) - chess.square_file(white_king_sq)) <= 1:
                if board.is_attacked_by(chess.BLACK, sq):
                    _set_plane_square(planes, 30, sq)
        planes[:, :, 32] = 1.0 if board.is_check() and board.turn == chess.WHITE else 0.0
        for sq in board.attacks(white_king_sq):
            piece = board.piece_at(sq)
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                _set_plane_square(planes, 34, sq)

    if black_king_sq is not None:
        for offset in king_zone_offsets:
            sq = black_king_sq + offset
            if 0 <= sq < 64 and abs(chess.square_file(sq) - chess.square_file(black_king_sq)) <= 1:
                if board.is_attacked_by(chess.WHITE, sq):
                    _set_plane_square(planes, 31, sq)
        planes[:, :, 33] = 1.0 if board.is_check() and board.turn == chess.BLACK else 0.0
        for sq in board.attacks(black_king_sq):
            piece = board.piece_at(sq)
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                _set_plane_square(planes, 35, sq)

    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
    white_files = [chess.square_file(sq) for sq in white_pawns]
    black_files = [chess.square_file(sq) for sq in black_pawns]
    white_file_set = set(white_files)
    black_file_set = set(black_files)
    for sq in white_pawns:
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        if file_idx - 1 not in white_file_set and file_idx + 1 not in white_file_set:
            _set_plane_square(planes, 38, sq)
        if white_files.count(file_idx) > 1:
            _set_plane_square(planes, 40, sq)
        blockers = [
            bp
            for bp in black_pawns
            if chess.square_file(bp) in (file_idx - 1, file_idx, file_idx + 1) and chess.square_rank(bp) > rank_idx
        ]
        if len(blockers) == 0:
            _set_plane_square(planes, 36, sq)

    for sq in black_pawns:
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        if file_idx - 1 not in black_file_set and file_idx + 1 not in black_file_set:
            _set_plane_square(planes, 39, sq)
        if black_files.count(file_idx) > 1:
            _set_plane_square(planes, 41, sq)
        blockers = [
            wp
            for wp in white_pawns
            if chess.square_file(wp) in (file_idx - 1, file_idx, file_idx + 1) and chess.square_rank(wp) < rank_idx
        ]
        if len(blockers) == 0:
            _set_plane_square(planes, 37, sq)

    for square, piece in board.piece_map().items():
        own_attackers = len(board.attackers(piece.color, square))
        opp_attackers = len(board.attackers(not piece.color, square))
        if opp_attackers > own_attackers:
            _set_plane_square(planes, 42 if piece.color == chess.WHITE else 43, square)
        if board.is_pinned(piece.color, square):
            _set_plane_square(planes, 44 if piece.color == chess.WHITE else 45, square)

    planes[:, :, 46] = np.clip((white_attackers_count - black_attackers_count) / 4.0, -1.0, 1.0)

    non_pawn_material = 0
    piece_values = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    for piece_type, value in piece_values.items():
        non_pawn_material += value * (
            len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
        )
    max_non_pawn_material = 62.0
    planes[:, :, 47] = min(1.0, non_pawn_material / max_non_pawn_material)

    center = np.array([3.5, 3.5], dtype=np.float32)
    enemy_king_sq = board.king(not board.turn)
    enemy_rc = (
        np.array(_square_row_col(enemy_king_sq), dtype=np.float32)
        if enemy_king_sq is not None
        else np.array([3.5, 3.5], dtype=np.float32)
    )
    for sq in chess.SQUARES:
        r, c = _square_row_col(sq)
        pos = np.array([r, c], dtype=np.float32)
        planes[r, c, 48] = np.linalg.norm(pos - center) / np.linalg.norm(np.array([0.0, 0.0]) - center)
        planes[r, c, 49] = np.linalg.norm(pos - enemy_rc) / np.linalg.norm(np.array([0.0, 0.0]) - center)

    planes[:, :, 50] = 1.0 if board.turn == chess.WHITE else -1.0
    for mv in board.legal_moves:
        _set_plane_square(planes, 51, mv.from_square)
        _set_plane_square(planes, 52, mv.to_square)
    for sq in board.checkers():
        _set_plane_square(planes, 53, sq)

    return planes


def tokens_to_feature_planes(tokens: np.ndarray) -> np.ndarray:
    return board_to_feature_planes(tokens_to_board(tokens))

def parse_board_response(response_data: dict) -> dict:
    """
    Parse the response from Go server's /getData endpoint.
    
    Args:
        response_data: Dictionary from JSON response containing X, Y, player, move, eval
    
    Returns:
        Dictionary with:
            - X: numpy array of 65 integers
            - Y: numpy array of 65 integers
            - player: string
            - move: string  
            - eval: float
    """
    # X and Y come as base64 encoded bytes or as byte arrays
    x_bytes = response_data['X']
    y_bytes = response_data['Y']
    
    # If they're base64 strings, decode them
    if isinstance(x_bytes, str):
        x_bytes = base64.b64decode(x_bytes)
        y_bytes = base64.b64decode(y_bytes)
    # If they're lists/arrays, convert to bytes
    elif isinstance(x_bytes, list):
        x_bytes = bytes(x_bytes)
        y_bytes = bytes(y_bytes)
    
    return {
        'X': bytes_to_tokens(x_bytes),
        'Y': bytes_to_tokens(y_bytes),
        'player': response_data['player'],
        'move': response_data['move'],
        'eval': float(response_data['eval'])
    }


# Token constants (matching Go constants)
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

TOKENS = {
    EMPTY_SQUARE: "empty",
    WHITE_PAWN: "white_pawn",
    WHITE_KNIGHT: "white_knight",
    WHITE_BISHOP: "white_bishop",
    WHITE_ROOK: "white_rook",
    WHITE_QUEEN: "white_queen",
    WHITE_KING: "white_king",
    BLACK_PAWN: "black_pawn",
    BLACK_KNIGHT: "black_knight",
    BLACK_BISHOP: "black_bishop",
    BLACK_ROOK: "black_rook",
    BLACK_QUEEN: "black_queen",
    BLACK_KING: "black_king",
    WHITE_TO_MOVE: "white_to_move",
    BLACK_TO_MOVE: "black_to_move"
}
