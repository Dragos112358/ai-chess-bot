# ♟️ AI Chess Bot

An AlphaZero-style chess engine: a **ResNet policy + value network** guides a **Monte Carlo Tree Search (PUCT)**. The network is trained with **PyTorch** on millions of Stockfish-evaluated positions, exported to **ONNX**, and playable **entirely in the browser** with no backend.

**[▶ Play it live](https://dragos112358.github.io/ai-chess-bot/)** · estimated strength **~2200–2700 Elo** on Stockfish's `UCI_Elo` scale, depending on the search budget (50 to 1600 MCTS simulations; see [Results](#results) for the caveats)

![screenshot](docs/screenshot.png)

---

## Highlights

- **Supervised training on Stockfish labels** from the Lichess evaluation database (policy = soft distribution over Stockfish's top-6 moves, value = squashed centipawn score)
- **Batched MCTS** with virtual loss, an evaluation cache, tree reuse, fp16 inference and early stopping
- **Runs in the browser**: `onnxruntime-web` with WebGPU and automatic WASM fallback, inference in a Web Worker so the UI never blocks
- **Measured, not guessed**: Elo estimated by match play against Stockfish levels 1800–3000, with search strength swept from 50 to 1600 simulations
- Full desktop app (pygame): play, prepare data, train, run an Elo arena, and duel a new model against the previous one

## Results

Matches against Stockfish limited with `UCI_Elo`, 0.1 s per move, **20 games per level** (colors alternated, random 4-ply openings shared by each pair of games), no opening book or tablebase. Each cell is the bot's **wins-draws-losses**. Every simulation budget was tested against Stockfish levels 1800 to 3000 in steps of 100 (`n/a` = skipped, because the bot scored under 5% at the previous level).

| Stockfish level | 50 sims | 100 sims | 200 sims | 400 sims | 800 sims | 1600 sims |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1800 | 12-4-4  | 12-3-5  | 11-5-4  | 12-4-4  | 18-1-1  | 19-0-1  |
| 1900 | 9-4-7   | 14-0-6  | 14-2-4  | 15-3-2  | 19-0-1  | 20-0-0  |
| 2000 | 8-6-6   | 10-3-7  | 9-5-6   | 12-4-4  | 15-3-2  | 19-1-0  |
| 2100 | 6-2-12  | 9-4-7   | 12-2-6  | 11-4-5  | 14-4-2  | 19-0-1  |
| 2200 | 4-4-12  | 6-6-8   | 9-2-9   | 9-5-6   | 15-2-3  | 18-1-1  |
| 2300 | 7-6-7   | 8-4-8   | 7-5-8   | 8-7-5   | 15-3-2  | 15-5-0  |
| 2400 | 5-4-11  | 2-6-12  | 6-3-11  | 8-5-7   | 9-9-2   | 14-4-2  |
| 2500 | 3-5-12  | 2-8-10  | 3-9-8   | 1-10-9  | 8-6-6   | 18-1-1  |
| 2600 | 1-5-14  | 1-7-12  | 3-8-9   | 2-10-8  | 9-5-6   | 9-4-7   |
| 2700 | 2-7-11  | 2-6-12  | 3-7-10  | 1-7-12  | 2-13-5  | 7-7-6   |
| 2800 | 0-5-15  | 0-4-16  | 0-8-12  | 2-4-14  | 1-7-12  | 5-8-7   |
| 2900 | 0-0-20  | 0-2-18  | 0-5-15  | 0-4-16  | 0-11-9  | 1-7-12  |
| 3000 | n/a     | 2-4-14  | 0-5-15  | 0-8-12  | 0-5-15  | 1-9-10  |
| **Estimated Elo** | **~2201** | **~2272** | **~2333** | **~2358** | **~2542** | **~2702** |

"Estimated Elo" is the average of the per-level performance estimates over the levels where the bot scored between 10% and 90% (levels outside that range say little about strength). The bot scores roughly 50% against SF ~2300 at 50–200 simulations, SF ~2400–2500 at 400–800, and SF ~2700–2800 at 1600.

**How to read this honestly**

- Strength clearly **grows with search budget**: going from 50 to 1600 simulations moves the 50%-score point from about SF 2200–2300 up to about SF 2700–2800.
- With 20 games per level the uncertainty is large (the 95% interval of a single level is roughly ±150 Elo), so individual cells are noisy. For example, 1600 simulations scored 93% against SF 2500 but only 55% against SF 2600. Compare the trend, not single cells.
- The per-level estimates **rise with the opponent's level** (e.g. at 800 simulations: ~2270 against SF 2000, ~2650 against SF 2600). If `UCI_Elo` were perfectly linear they would be flat, so the scale is clearly not linear at this time control. The levels where the score is closest to 50% are the most reliable.
- `UCI_Elo` is calibrated against Stockfish's own rating list at a much longer time control than 0.1 s per move. At 0.1 s Stockfish is probably weaker than its nominal rating, so these numbers likely **overstate** the bot's strength and are **not comparable to Lichess or FIDE ratings**.
- The measured numbers are for the **desktop engine**. The browser version uses a simpler search (no batching, cache or tree reuse) and offers up to 400 simulations, so expect somewhat different play at equal simulations.
- To tighten the estimate: 100+ games per level, focused on the levels near the 50% crossing point. `Find_ELO.py` is included for this.

## How it works

```
Lichess eval DB ─► labeled positions (data/*.npz) ─► PyTorch training ─► model.pt ─► export_onnx.py ─► model.onnx ─► browser
```

**Board encoding.** Each position is a `17 × 8 × 8` tensor from the point of view of the side to move (the board is flipped for black): 6 planes own pieces, 6 planes opponent pieces, 4 castling-rights planes, 1 en-passant plane. Positions are stored bit-packed (136 bytes each).

**Network.** A residual CNN (configurable depth and width; the default is 12 blocks × 192 channels) with two heads:
- **Policy:** logits over `64 × 64` (from-square × to-square), masked to legal moves.
- **Value:** a `tanh` scalar in `[-1, 1]` for the side to move.

**Training targets.** For each position the policy target is a softmax over Stockfish's top-6 moves (temperature 80 cp), and the value target is `tanh(cp / 400)` of the best line. The data pipeline can stream the Lichess evaluation database directly, or label PGN games with a local Stockfish.

**Training recipe.** AdamW with warmup and cosine decay, bf16 mixed precision, exponential moving average of the weights, left–right flip augmentation (only on positions without castling rights, where it is a valid symmetry), validation split **by game** to avoid leakage, and early stopping.

**Search.** PUCT with `c_puct = 1.5` and a first-play-urgency reduction of `0.15`. The desktop engine adds batched evaluation with virtual loss, a transposition cache, subtree reuse between moves, extra time when the top two moves are close, direct mate-in-1, and optional Polyglot opening book and Syzygy tablebase support. The browser version uses the same encoding and the same PUCT rule in a simpler, unbatched loop, with inference offloaded to a Web Worker.

## Browser app

- Play as white or black, search strength of 50 / 100 / 200 / 400 simulations
- Promotion picker, undo, evaluation bar (from the network's value head), legal-move hints, last-move highlight
- Full rules via `chess.js`: castling, en passant, check, mate, draws
- Mobile-friendly layout

## Run locally

The page loads `model.onnx` with `fetch`, so it has to be served over HTTP:

```bash
cd docs
python3 -m http.server 8000
```

Then open **http://localhost:8000**.

The desktop app (training, arena, duel):

```bash
pip install chess numpy pygame-ce zstandard torch
python Chess_bot.py
```

To measure Elo from a script:

```bash
python Find_ELO.py     # downloads Stockfish on first run, then plays the arena matches
```

## Project structure

```
.
├── Chess_bot.py      # desktop app: GUI, data prep, training, MCTS, arena, duel
├── Find_ELO.py       # Elo vs Stockfish for several simulation budgets
├── export_onnx.py    # PyTorch → ONNX export
├── chess_bot_settings.json
├── data/             # labeled position shards (.npz), not in the repo
├── checkpoints/      # model weights (.pt), not in the repo
└── docs/             # static web app served by GitHub Pages
    ├── index.html
    └── model.onnx
```

## Tech stack

Python · PyTorch · ONNX · onnxruntime-web (WebGPU / WASM) · Web Workers · JavaScript · chess.js · python-chess · pygame · GitHub Pages

## Limitations

- **No underpromotion for the bot:** the policy head predicts only from/to squares, so the bot always promotes to a queen (humans can pick any piece).
- **Supervised, not self-play:** the network imitates Stockfish's evaluations, so it inherits their strengths and biases; reinforcement learning from self-play is not implemented.
- **Browser vs desktop:** the browser engine is simpler (no batching, cache or tree reuse), so at equal simulations it can play differently from the measured desktop engine.
- The Elo estimate is noisy and relative to Stockfish's `UCI_Elo` scale at 0.1 s per move (see [Results](#results)).

## Roadmap

- [x] Move inference into a Web Worker so the UI never blocks
- [x] Undo button and evaluation bar
- [x] Elo measurement against Stockfish (levels 1800–3000, 50–1600 simulations)
- [ ] More games per level (100+) around the 50% crossing point to tighten the confidence intervals
- [ ] More data and a stronger network, then self-play fine-tuning
- [ ] Underpromotion support in the policy head