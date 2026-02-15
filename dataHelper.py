import numpy as np
import chess
import base64

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
