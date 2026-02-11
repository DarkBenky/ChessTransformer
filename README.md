# Model Architecture

## Next Board Model
- **Input:** Tokenized board + side to move
- **Output:** Next board
- **Data:** High‑rated Lichess games

## Board Eval Model
- **Input:** Tokenized board
- **Output:** Evaluation
- **Data:** High‑rated Lichess games evaluated by Stockfish

## Plan
1. Process data
2. Train Next Board model with Eval Model
3. Reinforcement training
4. Serve on Lichess

## Pipeline
Current board → Next Board model → generate $N$ boards (dropout enabled at inference)

Validate boards → evaluate predicted boards → explore highest‑eval boards more deeply

## Research
- Reinforcement learning to improve performance (ideas/approaches)
	- Fine‑tune with self‑play using a policy/value loss (AlphaZero‑style)
- Serving the model on Lichess from a VM (useful docs)
	- Lichess Bot API: https://lichess.org/api#tag/Bot
	- Lichess bot reference: https://github.com/lichess-bot-devs/lichess-bot
	- Lichess API docs: https://lichess.org/api
	- UCI protocol (engine interface): https://www.wbec-ridderkerk.nl/html/UCIProtocol.html