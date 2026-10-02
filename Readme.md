# ♟️ AI Chess Bot

A chess engine in the style of AlphaZero: a neural network (policy + value) guides a **Monte Carlo Tree Search**. Trained with **PyTorch** on self-play data, exported to **ONNX**, and playable **entirely in the browser**, with no backend.

**[▶ Play it live](https://github.com/Dragos112358/ai-chess-bot)**

<!-- Add a screenshot: save it as docs/screenshot.png and uncomment -->
[screenshot](docs/screenshot.png)

---

## Features

- Neural-network-guided MCTS (PUCT) with a policy head and a value head
- Runs 100% client-side: `onnxruntime-web` with **WebGPU**, automatic fallback to **WASM**
- Play as white or black, adjustable search strength (50 / 100 / 200 / 400 simulations)
- Full chess rules via `chess.js`: castling, en passant, promotion picker, check, mate, draws
- Move highlighting, legal-move hints, last-move marker
- Mobile-friendly layout

## How it works

```
self-play ──► data/*.npz ──► training (PyTorch) ──► model.pt ──► export_onnx.py ──► model.onnx ──► browser
```

**Board encoding.** The position is encoded as a `17 × 8 × 8` tensor, always from the point of view of the side to move (the board is flipped for black):

| Planes | Content |
|---|---|
| 0–5 | Own pieces: pawn, knight, bishop, rook, queen, king |
| 6–11 | Opponent pieces, same order |
| 12–13 | Own castling rights (king-side, queen-side) |
| 14–15 | Opponent castling rights |
| 16 | En passant target |

**Network output.**
- **Policy:** logits over `64 × 64` (from-square × to-square), masked to legal moves and softmax-normalized.
- **Value:** a scalar in `[-1, 1]` estimating the outcome for the side to move.

**Search.** Standard PUCT selection with `c_puct = 1.5` and a first-play-urgency reduction of `0.15`. The move with the most visits at the root is played. The same encoding and search logic is implemented in both Python (training) and JavaScript (browser), so the two behave identically.

## Run locally

The page loads `model.onnx` with `fetch`, so it must be served over HTTP (opening the file directly will not work):

```bash
cd docs
python3 -m http.server 8000
```

Then open **http://localhost:8000** (not `0.0.0.0`).

## Project structure

```
.
├── Chess_bot.py          # model, encoding, MCTS
├── main.py               # training / self-play entry point
├── Merge_files.py        # merges data shards
├── export_onnx.py        # PyTorch → ONNX export
├── chess_bot_settings.json
├── data/                 # self-play shards (.npz), not in the repo
├── checkpoints/          # model weights (.pt), not in the repo
└── docs/                 # static web app served by GitHub Pages
    ├── index.html
    └── model.onnx
```

## Tech stack

Python · PyTorch · ONNX · onnxruntime-web (WebGPU / WASM) · JavaScript · chess.js · GitHub Pages

## Limitations

- The bot always promotes to a queen: the policy head predicts only from/to squares, so underpromotion is not representable. (Humans can promote to any piece.)
- Strength depends on the amount of self-play training; more simulations make it stronger but slower, especially on the WASM fallback.

## Roadmap

- [ ] Move MCTS into a Web Worker so the UI never blocks
- [ ] More self-play iterations and a stronger network
- [ ] Underpromotion support in the policy head
- [ ] Undo button and evaluation bar
