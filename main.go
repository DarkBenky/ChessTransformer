package main

import (
	"bufio"
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	_ "github.com/mattn/go-sqlite3"
)

type BoardData struct {
	X      string `json:"X"`
	Y      string `json:"Y"`
	Player string `json:"player"`
	Move   string `json:"move"`
	GameID string `json:"game_id"`
}

const (
	emptySquare uint8 = iota
	whitePawn
	whiteKnight
	whiteBishop
	whiteRook
	whiteQueen
	whiteKing
	blackPawn
	blackKnight
	blackBishop
	blackRook
	blackQueen
	blackKing
	whiteToMove
	blackToMove
}

func initDB() error {
	var err error
	db, err = sql.Open("sqlite3", "file:/media/user/2TB Clear/chess_data.db?_busy_timeout=10000&_journal_mode=WAL&_synchronous=NORMAL")
	if err != nil {
		return err
	}

	// SQLite is safest with a single writer connection in this workload.
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)

	createTableSQL := `CREATE TABLE IF NOT EXISTS board_data (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		x_board BLOB NOT NULL,
		y_board BLOB NOT NULL,
		player TEXT NOT NULL,
		move TEXT NOT NULL,
		eval REAL NOT NULL,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);`

	_, err = db.Exec(createTableSQL)
	return err
}

var (
	db        *sql.DB
	dbWriteMu sync.Mutex
)

func fenToTokens(fen string) []byte {
	tokens := make([]byte, 65)
	idx := 1 // Start at index 1, reserve 0 for turn

	// Parse FEN parts
	parts := strings.Split(fen, " ")
	boardPart := parts[0]

	// Set turn token (first byte)
	if len(parts) > 1 && parts[1] == "b" {
		tokens[0] = blackToMove
	} else {
		tokens[0] = whiteToMove
	}

	// Parse board
	for _, char := range boardPart {
		if char == '/' {
			continue
		}

		if char >= '1' && char <= '8' {
			// Empty squares
			count := int(char - '0')
			for i := 0; i < count; i++ {
				if idx < 65 {
					tokens[idx] = emptySquare
					idx++
				}
			}
		} else {
			// Piece
			if idx < 65 {
				tokens[idx] = pieceToToken(byte(char))
				idx++
			}
		}
	}

	return tokens
}

func pieceToToken(piece byte) uint8 {
	switch piece {
	case 'P':
		return whitePawn
	case 'N':
		return whiteKnight
	case 'B':
		return whiteBishop
	case 'R':
		return whiteRook
	case 'Q':
		return whiteQueen
	case 'K':
		return whiteKing
	case 'p':
		return blackPawn
	case 'n':
		return blackKnight
	case 'b':
		return blackBishop
	case 'r':
		return blackRook
	case 'q':
		return blackQueen
	case 'k':
		return blackKing
	default:
		return emptySquare
	}
}

type EvalResult struct {
	BestMove string  `json:"best_move"`
	Eval     float64 `json:"eval"`
	Mate     *int    `json:"mate,omitempty"`
}

func analyzePosition(fen string, depth int) (*EvalResult, error) {
	cmd := exec.Command("stockfish")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start stockfish: %w", err)
	}
	defer cmd.Process.Kill()

	scanner := bufio.NewScanner(stdout)

	// Initialize UCI
	fmt.Fprintln(stdin, "uci")
	for scanner.Scan() {
		if strings.Contains(scanner.Text(), "uciok") {
			break
		}
	}

	// Set position
	fmt.Fprintf(stdin, "position fen %s\n", fen)

	// Start analysis
	fmt.Fprintf(stdin, "go depth %d\n", depth)

	var bestMove string
	var eval float64
	var mate *int

	for scanner.Scan() {
		line := scanner.Text()

		// Parse evaluation from info lines
		if strings.HasPrefix(line, "info") && strings.Contains(line, "score") {
			parts := strings.Fields(line)
			for i, part := range parts {
				if part == "score" && i+2 < len(parts) {
					scoreType := parts[i+1]
					scoreValue := parts[i+2]

					switch scoreType {
					case "cp":
						// Centipawn score
						if val, err := strconv.ParseFloat(scoreValue, 64); err == nil {
							eval = val / 100.0
						}
					case "mate":
						// Mate in X moves
						if val, err := strconv.Atoi(scoreValue); err == nil {
							mate = &val
							eval = 0
						}
					}
				}
			}
		}

		// Parse best move
		if strings.HasPrefix(line, "bestmove") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				bestMove = parts[1]
			}
			break
		}
	}

	fmt.Fprintln(stdin, "quit")

	if bestMove == "" {
		return nil, fmt.Errorf("no best move found")
	}

	return &EvalResult{
		BestMove: bestMove,
		Eval:     eval,
		Mate:     mate,
	}, nil
}

func applyMove(fen string, move string) (string, error) {
	cmd := exec.Command("stockfish")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return "", fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start stockfish: %w", err)
	}
	defer cmd.Process.Kill()

	scanner := bufio.NewScanner(stdout)

	// Initialize UCI
	fmt.Fprintln(stdin, "uci")
	for scanner.Scan() {
		if strings.Contains(scanner.Text(), "uciok") {
			break
		}
	}

	// Set position and make move
	fmt.Fprintf(stdin, "position fen %s moves %s\n", fen, move)
	fmt.Fprintln(stdin, "d")

	var resultFen string
	inFenSection := false

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "Fen:") {
			inFenSection = true
			resultFen = strings.TrimPrefix(line, "Fen: ")
			break
		}
	}

	fmt.Fprintln(stdin, "quit")

	if !inFenSection || resultFen == "" {
		return "", fmt.Errorf("failed to get FEN after move")
	}

	return resultFen, nil
}

func main() {
	if err := initDB(); err != nil {
		log.Fatal("Failed to initialize database:", err)
	}
	defer db.Close()

	log.Println("Database initialized successfully")

	e := echo.New()

	// Disable request logging for cleaner output
	// e.Use(middleware.Logger())
	e.Use(middleware.Recover())

	e.POST("/postBoard", postBoardHandler)
	e.GET("/getData", getBoard)

	e.Logger.Fatal(e.Start(":1323"))
}

func postBoardHandler(c echo.Context) error {
	data := new(BoardData)

	if err := c.Bind(data); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "Invalid request"})
	}

	// Process inline to keep behavior simple and avoid hidden goroutine crashes.
	xTokens := fenToTokens(data.X)
	yTokens := fenToTokens(data.Y)

	evalResult, err := analyzePosition(data.X, 8)
	if err != nil {
		log.Printf("Failed to analyze: %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": "analysis failed"})
	}

	predictedFen, err := applyMove(data.X, evalResult.BestMove)
	if err != nil {
		log.Printf("Failed to apply move: %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": "apply move failed"})
	}

	yPredictedTokens := fenToTokens(predictedFen)

	dbWriteMu.Lock()
	defer dbWriteMu.Unlock()

	_, err = db.Exec(
		"INSERT INTO board_data (x_board, y_board, player, move, eval) VALUES (?, ?, ?, ?, ?)",
		xTokens, yTokens, data.Player, data.Move, evalResult.Eval,
	)
	if err != nil {
		log.Printf("DB error (human): %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": "db insert failed"})
	}

	_, err = db.Exec(
		"INSERT INTO board_data (x_board, y_board, player, move, eval) VALUES (?, ?, ?, ?, ?)",
		xTokens, yPredictedTokens, "stockfish", evalResult.BestMove, evalResult.Eval,
	)
	if err != nil {
		log.Printf("DB error (stockfish): %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": "db insert failed"})
	}

	log.Printf("Inserted moves: human=%s stockfish=%s", data.Move, evalResult.BestMove)
	return c.JSON(http.StatusOK, map[string]string{"status": "stored"})
}

func getBoard(c echo.Context) error {
	// return random row from database
	row := db.QueryRow("SELECT x_board, y_board, player, move, eval FROM board_data ORDER BY RANDOM() LIMIT 1")

	var xBoard []byte
	var yBoard []byte
	var player string
	var move string
	var eval float64

	err := row.Scan(&xBoard, &yBoard, &player, &move, &eval)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": "Failed to fetch data"})
	}

	return c.JSON(http.StatusOK, map[string]interface{}{
		"X":      xBoard,
		"Y":      yBoard,
		"player": player,
		"move":   move,
		"eval":   eval,
	})
}
