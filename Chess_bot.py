"""
chess_bot.py - botul de sah cu interfata grafica, TOT intr-un singur fisier (v4).

Pornire:   python chess_bot.py        (fara argumente; totul se face din butoane)

Instalare: pip install chess numpy pygame-ce zstandard   (+ PyTorch cu CUDA: buton "Fix GPU" in meniu)

Ce e nou fata de v3:
  - date: baza de evaluari Lichess (sute de milioane de pozitii deja evaluate de Stockfish, fara Stockfish local)
  - validare impartita pe PARTIDE (fara scurgere de date), augmentare prin oglindire, date mari tinute pe CPU daca nu incap pe GPU
  - MCTS mai rapid: encodare cu bitboard-uri, cache de evaluari, reutilizarea arborelui, inferenta in fp16, oprire timpurie
  - carte de deschideri (Polyglot), tablebase Syzygy, mat in 1 direct
  - arena: deschideri aleatorii pe perechi de partide, interval de incredere, duel "model nou vs model vechi"
"""
import copy
import glob
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine
import chess.pgn
import chess.polyglot
import chess.syzygy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pygame
except ImportError:
    pygame = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)                      # data/ si checkpoints/ stau langa script

# ======================================================================
# SETARI (se pot schimba si din aplicatie: butonul "Setari"; se salveaza automat)
# ======================================================================
DEFAULT_PGN_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
DEFAULT_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
DATA_DIR = "data"
CHECKPOINT = "checkpoints/model.pt"
PREVIOUS = "checkpoints/previous.pt"          # modelul de dinainte de ultima antrenare (pentru duel)
SETTINGS_FILE = "chess_bot_settings.json"
SHARD_SIZE = 250_000
MULTIPV = 6                   # cate mutari "bune" pastram pe pozitie (tinta policy)
ARENA_MOVETIME = 0.1
ARENA_OPENING_PLIES = 4       # deschidere aleatorie (comuna perechii de partide, culori inversate)
CFG_VERSION = 4
POLICY_TEMP = 80.0            # centipioni: cat de "moale" e distributia tinta peste cele mai bune mutari
VALUE_SF_WEIGHT = 0.75        # (doar sursa PGN) tinta value = 75% evaluarea Stockfish + 25% rezultatul partidei
PREP_CHUNK = 64
PATIENCE = 4                  # opreste antrenarea daca validarea nu se imbunatateste N epoci
VAL_MOD = 40                  # ~1/40 din PARTIDE merg in validare (dupa id-ul partidei)
CACHE_MAX = 150_000           # intrari in cache-ul de evaluari al MCTS


def default_workers():
    return max(1, min(24, (os.cpu_count() or 4) - 2))


CFG = dict(
    cfg_version=CFG_VERSION,
    data_source="evaldb",       # "evaldb" = baza de evaluari Lichess | "pgn" = partide PGN + Stockfish local
    eval_src=DEFAULT_EVAL_URL,  # URL sau cale catre lichess_db_eval.jsonl(.zst)
    pgn=DEFAULT_PGN_URL,        # URL sau cale catre un .pgn / .pgn.zst
    min_elo=1800,               # (pgn) doar partide in care AMBII jucatori au cel putin acest ELO
    target_positions=5_000_000,
    pos_per_game=10,            # (pgn) cate pozitii diferite se iau dintr-o partida
    sf_depth=12,                # (pgn) adancimea Stockfish la etichetare
    workers=default_workers(),  # (pgn) cate procese Stockfish ruleaza in paralel
    append_data=False,          # True = adauga la datele existente in loc sa le stearga
    epochs=20, batch=1024, blocks=12, channels=192, lr=1.5e-3,
    resume=False,               # True = continua antrenarea din modelul salvat
    sims=800,                   # simulari MCTS pe mutare
    play_white=True,
    stockfish=r"C:\stockfish\stockfish.exe",
    stockfish_elo=None,         # None = putere maxima
    arena_games=40,
    arena_elos=[1400, 1600, 1800],
    book="",                    # fisier Polyglot (.bin), optional
    syzygy="",                  # folder cu tablebase Syzygy, optional
    use_extras=True,            # foloseste cartea de deschideri / tablebase cand sunt setate
    cuda_tag="cu128",
)


def load_cfg():
    """Setarile vechi se pastreaza (doar cheile cunoscute); cele noi iau valorile implicite."""
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k in CFG and k != "cfg_version":
                CFG[k] = v
    except Exception:
        pass
    CFG["cfg_version"] = CFG_VERSION


def save_cfg():
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(CFG, f, indent=2)
    except Exception:
        pass


# ======================================================================
# CODIFICARE TABLA / MUTARI  (neschimbata fata de v3: datele si modelele vechi raman compatibile)
# ======================================================================
NUM_PLANES = 17                              # 12 piese + 4 rocade + 1 en passant
POLICY_SIZE = 64 * 64                        # index mutare = from * 64 + to


def _sq(sq: int, turn: bool) -> int:
    return sq ^ 56 if turn == chess.BLACK else sq


def board_to_planes(board: chess.Board) -> np.ndarray:
    """(17, 8, 8) uint8, din perspectiva celui la mutare (0-5 piesele lui, 6-11 ale adversarului).
    Piesele se iau direct din bitboard-uri (mult mai rapid decat parcurgerea piece_map)."""
    us = board.turn
    x = np.zeros((NUM_PLANES, 8, 8), dtype=np.uint8)
    buf = b"".join(int(board.pieces_mask(pt, c)).to_bytes(8, "little")
                   for c in (us, not us) for pt in range(1, 7))
    x[:12] = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little").reshape(12, 8, 8)
    if us == chess.BLACK:
        x[:12] = x[:12, ::-1, :].copy()                  # tabla rasturnata pe verticala pentru negru
    if board.has_kingside_castling_rights(us):
        x[12] = 1
    if board.has_queenside_castling_rights(us):
        x[13] = 1
    if board.has_kingside_castling_rights(not us):
        x[14] = 1
    if board.has_queenside_castling_rights(not us):
        x[15] = 1
    if board.ep_square is not None:
        s = _sq(board.ep_square, us)
        x[16, s >> 3, s & 7] = 1
    return x


def pack_planes(x: np.ndarray) -> np.ndarray:
    return np.packbits(x.reshape(-1))        # 136 x uint8


def move_index(move: chess.Move, turn: bool) -> int:
    return _sq(move.from_square, turn) * 64 + _sq(move.to_square, turn)


def legal_moves_with_index(board: chess.Board):
    """[(move, index)]; subpromovarile (N/B/R) sunt ignorate, promovarea e mereu dama."""
    out = []
    for m in board.legal_moves:
        if m.promotion and m.promotion != chess.QUEEN:
            continue
        out.append((m, move_index(m, board.turn)))
    return out


# ======================================================================
# RETEAUA NEURONALA
# ======================================================================
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class ChessNet(nn.Module):
    def __init__(self, blocks=10, channels=128):
        super().__init__()
        self.blocks, self.channels = blocks, channels
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 64, POLICY_SIZE))
        self.value = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        h = self.tower(self.stem(x))
        return self.policy(h), self.value(h).squeeze(-1)


def save_checkpoint(path, model, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = {"model": model.state_dict(), "blocks": model.blocks, "channels": model.channels}
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_checkpoint(path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    model = ChessNet(d["blocks"], d["channels"]).to(device)
    model.load_state_dict(d["model"])
    return model, d


# ======================================================================
# TASKURI IN FUNDAL (pregatire date / antrenare / arena) - ruleaza in thread, GUI-ul ramane fluid
# ======================================================================
class Task:
    def __init__(self, fn, title):
        self.fn, self.title = fn, title
        self.lines, self.series = [], {}
        self.progress, self.status = 0.0, ""
        self.parts = 1                      # (arena/duel) in cate parti se imparte bara de progres
        self.stop = self.done = self.failed = False

    def log(self, msg):
        print(msg)
        self.lines.append(str(msg))
        if len(self.lines) > 400:
            del self.lines[:100]

    def point(self, name, y):
        self.series.setdefault(name, []).append(float(y))

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self.fn(self)
        except Exception as e:
            self.failed = True
            self.log(f"EROARE: {e}")
            for ln in traceback.format_exc().splitlines()[-4:]:
                self.log(ln)
        finally:
            self.done = True


# ---------------- pregatire date: componente comune ----------------
RE_W = re.compile(r'\[WhiteElo "(\d+)"\]')
RE_B = re.compile(r'\[BlackElo "(\d+)"\]')
RE_RES = re.compile(r'\[Result "([^"]+)"\]')
RE_TERM = re.compile(r'\[Termination "([^"]+)"\]')
RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def open_stream(src):
    """Deschide un fisier local SAU un URL (streaming: se citeste doar cat e nevoie). Suporta .zst."""
    if src.startswith("http"):
        import urllib.request
        req = urllib.request.Request(src, headers={"User-Agent": "chess-bot/1.0"})
        raw = urllib.request.urlopen(req, timeout=60)
    else:
        raw = open(src, "rb")
    if src.endswith(".zst"):
        import zstandard
        raw = zstandard.ZstdDecompressor().stream_reader(raw)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def iter_games(stream):
    buf = []
    for line in stream:
        if line.startswith("[Event ") and buf:
            yield "".join(buf)
            buf = []
        buf.append(line)
    if buf:
        yield "".join(buf)


class ShardWriter:
    """Aduna pozitii si le scrie in shard-uri .npz (x, pi, pp, v, g)."""
    def __init__(self, t, append):
        self.t = t
        os.makedirs(DATA_DIR, exist_ok=True)
        existing = sorted(glob.glob(os.path.join(DATA_DIR, "shard_*.npz")))
        self.id = 0
        if append and existing:
            self.id = int(re.findall(r"(\d+)", os.path.basename(existing[-1]))[-1]) + 1
            t.log(f"Adaug la {len(existing)} shard-uri existente.")
        else:
            for f in existing:
                os.remove(f)
        self.buf, self.n, self.total = [], 0, 0

    def add(self, x, pi, pp, v, g):
        self.buf.append((x, pi, pp, v, g))
        self.n += len(x)
        self.total += len(x)
        if self.n >= SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        path = os.path.join(DATA_DIR, f"shard_{self.id:04d}.npz")
        np.savez(path, x=np.concatenate([b[0] for b in self.buf]), pi=np.concatenate([b[1] for b in self.buf]),
                 pp=np.concatenate([b[2] for b in self.buf]), v=np.concatenate([b[3] for b in self.buf]),
                 g=np.concatenate([b[4] for b in self.buf]))
        self.t.log(f"salvat {path} ({self.n:,} pozitii)")
        self.buf, self.n = [], 0
        self.id += 1


def soft_policy(cps, idxs):
    """cps (centipioni, din perspectiva celui la mutare) + indici mutari -> (pi, pp) de lungime MULTIPV,
    sortate descrescator (pi[0] = cea mai buna mutare)."""
    cps = np.array(cps, dtype=np.float64)
    order = np.argsort(-cps)[:MULTIPV]
    cps = cps[order]
    idxs = [idxs[i] for i in order]
    c = np.clip(cps, -1500, 1500) / POLICY_TEMP
    c -= c.max()
    p = np.exp(c)
    p /= p.sum()
    pi = np.full(MULTIPV, -1, dtype=np.int16)
    pp = np.zeros(MULTIPV, dtype=np.float32)
    pi[:len(idxs)] = idxs
    pp[:len(idxs)] = p
    return pi, pp, float(cps[0])


# ---------------- sursa 1: baza de evaluari Lichess (fara Stockfish local) ----------------
def _material_diff(board):
    d = 0
    for pt, val in PIECE_VAL.items():
        d += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return d


def detect_white_relative(lines):
    """Stabileste automat daca scorurile "cp" din baza de evaluari sunt din perspectiva albului sau a celui la mutare:
    la pozitiile cu NEGRU la mutare si avantaj clar de material, semnul scorului arata conventia."""
    a = b = 0
    for line in lines:
        try:
            d = json.loads(line)
            ev = max(d["evals"], key=lambda e: e.get("depth", 0))
            cp = ev["pvs"][0].get("cp")
            if cp is None or abs(cp) < 150:
                continue
            board = chess.Board(d["fen"])
            if board.turn != chess.BLACK:
                continue
            md = _material_diff(board)
            if abs(md) < 3:
                continue
            if (cp > 0) == (md > 0):
                a += 1          # cp pozitiv = alb mai bine -> perspectiva albului
            else:
                b += 1          # cp pozitiv = negru (la mutare) mai bine -> perspectiva celui la mutare
        except Exception:
            continue
    return a >= b, a, b


def parse_eval_line(line, white_rel):
    """O linie din lichess_db_eval.jsonl -> (x_packed, pi, pp, v) sau None."""
    d = json.loads(line)
    evals = d.get("evals")
    if not evals:
        return None
    ev = max(evals, key=lambda e: e.get("depth", 0))
    pvs = ev.get("pvs") or []
    if not pvs:
        return None
    board = chess.Board(d["fen"])
    turn = board.turn
    sign = -1 if (white_rel and turn == chess.BLACK) else 1
    cps, idxs = [], []
    for pv in pvs[:MULTIPV + 4]:
        ln = pv.get("line")
        if not ln:
            continue
        uci = ln.split(" ", 1)[0]
        if len(uci) == 5 and uci[4] != "q":
            continue                                   # subpromovare: ignorata
        if len(uci) not in (4, 5):
            continue
        cp, mate = pv.get("cp"), pv.get("mate")
        if cp is None:
            if mate is None:
                continue
            cp = 10000 if mate > 0 else -10000
        try:
            frm, to = chess.parse_square(uci[:2]), chess.parse_square(uci[2:4])
        except ValueError:
            continue
        cps.append(cp * sign)
        idxs.append(_sq(frm, turn) * 64 + _sq(to, turn))
    if not idxs:
        return None
    pi, pp, best = soft_policy(cps, idxs)
    if abs(best) > 900 and random.random() > 0.3:      # prea multe pozitii deja decise strica echilibrul
        return None
    v = math.tanh(max(-2000.0, min(2000.0, best)) / 400.0)
    return pack_planes(board_to_planes(board)), pi, pp, v


def task_prepare_evaldb(t: Task):
    src = (CFG["eval_src"] or DEFAULT_EVAL_URL).strip()
    if not src.startswith("http") and not os.path.exists(src):
        raise RuntimeError(f"Nu gasesc fisierul: {src}  (alege-l din Setari)")
    target = CFG["target_positions"]
    writer = ShardWriter(t, CFG["append_data"])
    t.log(f"Sursa: {src}")
    t.log(f"Citesc evaluari Stockfish deja calculate; tinta: {target:,} pozitii.")
    xs, pis, pps, vs = [], [], [], []
    t0, seen, labeled = time.time(), 0, 0
    err = None

    def push():
        nonlocal xs, pis, pps, vs
        if not xs:
            return
        g = np.array([random.getrandbits(31) for _ in xs], dtype=np.int32)    # fiecare pozitie = "partida" proprie
        writer.add(np.stack(xs), np.stack(pis), np.stack(pps).astype(np.float16), np.array(vs, dtype=np.float16), g)
        xs, pis, pps, vs = [], [], [], []

    try:
        stream = open_stream(src)
        head = list(itertools.islice(stream, 3000))
        white_rel, a, b = detect_white_relative(head)
        t.log(f"Conventie scor detectata: {'perspectiva albului' if white_rel else 'perspectiva celui la mutare'} (voturi {a} vs {b}).")
        for line in itertools.chain(head, stream):
            if t.stop or labeled >= target:
                break
            seen += 1
            try:
                row = parse_eval_line(line, white_rel)
            except Exception:
                continue
            if row is None:
                continue
            xs.append(row[0])
            pis.append(row[1])
            pps.append(row[2])
            vs.append(row[3])
            labeled += 1
            if len(xs) >= 4096:
                push()
            if seen % 2000 == 0:
                rate = labeled / max(1e-6, time.time() - t0)
                eta = (target - labeled) / rate if rate > 0 else 0
                t.progress = min(1.0, labeled / target)
                t.status = f"pozitii {labeled:,}/{target:,} | {rate:.0f} poz/s | {seen:,} linii citite | ramas ~{eta / 60:.0f} min"
    except Exception as ex:
        err = ex
    push()
    writer.flush()
    if err:
        raise err
    t.progress = 1.0
    t.log(f"{'OPRIT' if t.stop else 'GATA'}: {writer.total:,} pozitii in {(time.time() - t0) / 60:.1f} min. Acum poti antrena.")


# ---------------- sursa 2: partide PGN + Stockfish local ----------------
_tls = threading.local()
_engines, _engines_lock = [], threading.Lock()


def _get_engine():
    """Un proces Stockfish per thread (Threads=1): paralelism real pe toate nucleele."""
    eng = getattr(_tls, "eng", None)
    if eng is None:
        eng = chess.engine.SimpleEngine.popen_uci(CFG["stockfish"])
        try:
            eng.configure({"Threads": 1, "Hash": 32})
        except Exception:
            pass
        _tls.eng = eng
        with _engines_lock:
            _engines.append(eng)
    return eng


def _close_engines():
    with _engines_lock:
        for e in _engines:
            try:
                e.quit()
            except Exception:
                pass
        _engines.clear()


def label_chunk(chunk, use_sf, depth):
    """chunk: [(fen, rezultat_alb, index_mutare_umana, id_partida)] -> (x_packed, pi, pp, v, g) sau None.
    v = evaluarea Stockfish (tanh) amestecata cu rezultatul real al partidei."""
    eng = _get_engine() if use_sf else None
    xs, pis, pps, vs, gs = [], [], [], [], []
    for fen, res_white, human_idx, gid in chunk:
        board = chess.Board(fen)
        legal = {m: i for m, i in legal_moves_with_index(board)}
        if not legal or board.is_game_over():
            continue
        res = res_white if board.turn == chess.WHITE else -res_white
        if eng is not None:
            infos = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=MULTIPV)
            cps, idxs = [], []
            for info in infos:
                pv, sc = info.get("pv"), info.get("score")
                if not pv or sc is None or pv[0] not in legal:
                    continue
                cps.append(sc.relative.score(mate_score=10000))
                idxs.append(legal[pv[0]])
            if not idxs:
                continue
            pi, pp, best = soft_policy(cps, idxs)
            best = max(-2000, min(2000, best))
            v = VALUE_SF_WEIGHT * math.tanh(best / 400.0) + (1 - VALUE_SF_WEIGHT) * res
        else:
            if human_idx < 0:
                continue
            pi = np.full(MULTIPV, -1, dtype=np.int16)
            pp = np.zeros(MULTIPV, dtype=np.float32)
            pi[0], pp[0], v = human_idx, 1.0, float(res)
        xs.append(pack_planes(board_to_planes(board)))
        pis.append(pi)
        pps.append(pp)
        vs.append(v)
        gs.append(gid)
    if not xs:
        return None
    return (np.stack(xs), np.stack(pis), np.stack(pps).astype(np.float16),
            np.array(vs, dtype=np.float16), np.array(gs, dtype=np.int32))


def task_prepare_pgn(t: Task):
    src = (CFG["pgn"] or DEFAULT_PGN_URL).strip()
    if not src.startswith("http") and not os.path.exists(src):
        raise RuntimeError(f"Nu gasesc fisierul: {src}  (alege-l din Setari)")
    min_elo, target, per_game, depth = CFG["min_elo"], CFG["target_positions"], CFG["pos_per_game"], CFG["sf_depth"]
    use_sf = os.path.exists(CFG["stockfish"])
    workers = CFG["workers"] if use_sf else 1
    if use_sf:
        try:
            e = chess.engine.SimpleEngine.popen_uci(CFG["stockfish"])
            e.quit()
        except Exception as ex:
            raise RuntimeError(f"Nu pot porni Stockfish ({ex}). Alege alt fisier din Setari.")
        t.log(f"Stockfish: {CFG['stockfish']}")
        t.log(f"Etichetez {target:,} pozitii, adancime {depth}, {workers} fire in paralel.")
    else:
        t.log("ATENTIE: nu am gasit Stockfish -> folosesc doar mutarile oamenilor (calitate MULT mai mica).")
        t.log("ATENTIE: mai bine foloseste sursa 'Eval Lichess' din Setari (nu cere Stockfish).")

    writer = ShardWriter(t, CFG["append_data"])
    t.log(f"Sursa pozitii: {src}")

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = deque()
    labeled = submitted = games_used = games_seen = 0
    queue, seen_keys = [], set()
    max_pending = workers * 3
    t0 = time.time()
    err = None

    def take(res):
        nonlocal labeled
        if res is None:
            return
        writer.add(*res)
        labeled += len(res[0])

    def collect():
        while futures and (len(futures) > max_pending or futures[0].done()):
            take(futures.popleft().result())
        rate = labeled / max(1e-6, time.time() - t0)
        eta = (target - labeled) / rate if rate > 0 and labeled else 0
        t.progress = min(1.0, labeled / target)
        t.status = f"etichetate {labeled:,}/{target:,} | {rate:.0f} poz/s | {games_seen:,} partide citite | ramas ~{eta / 60:.0f} min"

    try:
        for text in iter_games(open_stream(src)):
            if t.stop or submitted >= target:
                break
            games_seen += 1
            mw, mb, mr = RE_W.search(text), RE_B.search(text), RE_RES.search(text)
            if not (mw and mb and mr) or min(int(mw.group(1)), int(mb.group(1))) < min_elo:
                continue
            if mr.group(1) not in RESULTS:
                continue
            mt = RE_TERM.search(text)
            if mt and mt.group(1) != "Normal":
                continue
            result_white = RESULTS[mr.group(1)]
            try:
                game = chess.pgn.read_game(io.StringIO(text))
                if game is None:
                    continue
                moves = list(game.mainline_moves())
                if len(moves) < 14:
                    continue
                gid = random.getrandbits(31)
                chosen = set(random.sample(range(6, len(moves)), min(per_game, len(moves) - 6)))
                board = game.board()
                for ply, mv in enumerate(moves):
                    if ply in chosen:
                        key = hash(board.epd())
                        if key not in seen_keys:
                            seen_keys.add(key)
                            hidx = -1 if (mv.promotion and mv.promotion != chess.QUEEN) else move_index(mv, board.turn)
                            queue.append((board.fen(), result_white, hidx, gid))
                    board.push(mv)
            except Exception:
                continue
            games_used += 1
            if len(queue) >= PREP_CHUNK:
                futures.append(pool.submit(label_chunk, list(queue), use_sf, depth))
                submitted += len(queue)
                queue = []
                collect()
        if queue and not t.stop:
            futures.append(pool.submit(label_chunk, list(queue), use_sf, depth))
            submitted += len(queue)
    except Exception as ex:
        err = ex
    stopped = t.stop or err is not None
    pool.shutdown(wait=True, cancel_futures=stopped)
    for f in list(futures):
        if not f.cancelled():
            take(f.result())
    futures.clear()
    collect()
    writer.flush()
    _close_engines()
    if err:
        raise err
    t.progress = 1.0
    t.log(f"{'OPRIT' if t.stop else 'GATA'}: {labeled:,} pozitii din {games_used:,} partide in {(time.time() - t0) / 60:.1f} min. Acum poti antrena.")


def task_prepare(t: Task):
    if CFG["data_source"] == "evaldb":
        task_prepare_evaldb(t)
    else:
        task_prepare_pgn(t)


# ---------------- antrenare ----------------
_SHIFTS = {}


def unpack(xb):
    """(B,136) uint8 -> (B,17,8,8)."""
    s = _SHIFTS.get(xb.device)
    if s is None:
        s = _SHIFTS[xb.device] = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.uint8, device=xb.device)
    bits = (xb.unsqueeze(-1) >> s) & 1
    return bits.reshape(xb.shape[0], NUM_PLANES, 8, 8)


def load_data(folder):
    files = sorted(glob.glob(os.path.join(folder, "shard_*.npz")))
    if not files:
        raise RuntimeError("Nu am date de antrenare. Ruleaza intai 'Pregateste datele'.")
    parts = []
    for f in files:
        d = np.load(f)
        n = len(d["x"])
        g = d["g"].astype(np.int32) if "g" in d.files else np.random.randint(0, 2 ** 31 - 1, n, dtype=np.int32)
        if "pi" in d.files:
            parts.append((d["x"], d["pi"], d["pp"], d["v"].astype(np.float32), g))
        else:                                                  # format vechi (doar mutari umane)
            p = d["p"].astype(np.int16)[:, None]
            parts.append((d["x"], p, np.ones((len(p), 1), dtype=np.float16), d["v"].astype(np.float32), g))
    K = max(p[1].shape[1] for p in parts)

    def pad(a, fill, dtype):
        if a.shape[1] == K:
            return a
        out = np.full((len(a), K), fill, dtype=dtype)
        out[:, :a.shape[1]] = a
        return out

    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([pad(p[1], -1, np.int16) for p in parts]),
            np.concatenate([pad(p[2], 0, np.float16) for p in parts]),
            np.concatenate([p[3] for p in parts]),
            np.concatenate([p[4] for p in parts]))


class Data:
    """Tine datele pe GPU daca incap (rapid); altfel pe CPU, iar loturile se muta pe GPU la nevoie."""
    def __init__(self, arrays, device):
        X, PI, PP, V, G = arrays
        self.n, self.device, self.G = len(X), device, G
        need = X.nbytes + PI.nbytes + PP.nbytes + V.nbytes // 2
        free = torch.cuda.mem_get_info()[0] if device.type == "cuda" else 0
        self.on_gpu = device.type == "cuda" and need < 0.55 * free
        self.idev = device if self.on_gpu else torch.device("cpu")
        mv = (lambda a: torch.from_numpy(a).to(device)) if self.on_gpu else torch.from_numpy
        self.X, self.PI, self.PP = mv(X), mv(PI), mv(PP)
        self.V = mv(V.astype(np.float16))

    def get(self, idx):
        x, pi, pp, v = self.X[idx], self.PI[idx], self.PP[idx], self.V[idx]
        if not self.on_gpu:
            x, pi, pp, v = (a.to(self.device) for a in (x, pi, pp, v))
        return x, pi.long(), pp.float(), v.float()


def split_by_game(G):
    """Validarea se alege dupa id-ul PARTIDEI, deci pozitii din aceeasi partida nu ajung in ambele seturi."""
    val_mask = (G % VAL_MOD) == 0
    val_all, tr = np.nonzero(val_mask)[0], np.nonzero(~val_mask)[0]
    leak = False
    if len(val_all) < 2000:                                    # prea putine date: impartire simpla (cu risc de scurgere)
        perm = np.random.permutation(len(G))
        k = min(50_000, max(200, len(G) // 40))
        val_all, tr, leak = perm[:k], perm[k:], True
    val = np.random.permutation(val_all)[:50_000]
    return tr, val, leak


def policy_loss(logits, pi, pp):
    """Cross-entropy cu tinta "moale": distributie peste top-K mutari Stockfish."""
    logp = F.log_softmax(logits.float(), dim=1)
    g = logp.gather(1, pi.clamp(min=0))
    return -(pp * g).sum(1).mean()


def augment_flip(xb, pi):
    """Oglindire stanga-dreapta, valida doar cand NU exista drepturi de rocada (sah e simetric doar atunci)."""
    no_castle = xb[:, 12:16].amax(dim=(1, 2, 3)) == 0
    do = no_castle & (torch.rand(len(xb), device=xb.device) < 0.5)
    if not bool(do.any()):
        return xb, pi
    xb = torch.where(do.view(-1, 1, 1, 1), xb.flip(3), xb)
    flipped = (((pi >> 6) ^ 7) << 6) | ((pi & 63) ^ 7)
    pi = torch.where(do.view(-1, 1) & (pi >= 0), flipped, pi)
    return xb, pi


@torch.no_grad()
def evaluate(model, data, idx, device, use_amp, amp_dtype, bs=4096):
    was_training = model.training
    model.eval()
    ok, lp_sum, mse, n = 0, 0.0, 0.0, 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x, pi, pp, v_t = data.get(b)
        xb = unpack(x).float()
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits, v = model(xb)
        ok += (logits.argmax(1) == pi[:, 0]).sum().item()
        lp_sum += policy_loss(logits, pi, pp).item() * len(b)
        mse += F.mse_loss(v.float(), v_t, reduction="sum").item()
        n += len(b)
    model.train(was_training)
    return ok / n, lp_sum / n, mse / n


def task_train(t: Task):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        t.log(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        t.log("ATENTIE: PyTorch nu vede GPU-ul -> antrenez pe CPU (FOARTE lent). Din meniu apasa 'Fix GPU'.")
    use_amp = cuda
    amp_dtype = torch.bfloat16 if (not cuda or torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(cuda and amp_dtype == torch.float16))

    arrays = load_data(DATA_DIR)
    data = Data(arrays, device)
    n = data.n
    t.log(f"{n:,} pozitii incarcate (top-{arrays[1].shape[1]} mutari / pozitie), date pe {'GPU' if data.on_gpu else 'CPU (nu incap pe GPU)'}")
    tr_np, val_np, leak = split_by_game(data.G)
    if leak:
        t.log("ATENTIE: prea putine partide pentru o validare curata; validarea poate fi usor optimista.")
    tr_idx = torch.from_numpy(tr_np).to(data.idev)
    val_idx = torch.from_numpy(val_np).to(data.idev)
    t.log(f"antrenare: {len(tr_idx):,} | validare: {len(val_idx):,} (impartit pe partide)")

    had_model = os.path.exists(CHECKPOINT)
    if had_model:
        try:
            shutil.copy(CHECKPOINT, PREVIOUS)               # pentru duelul "model nou vs vechi"
        except Exception:
            pass
    if had_model and CFG.get("resume", False):
        model, _ = load_checkpoint(CHECKPOINT, device)
        t.log("Continui din modelul salvat.")
    else:
        model = ChessNet(CFG["blocks"], CFG["channels"]).to(device)
    if cuda:
        model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model)                       # media mobila a ponderilor: model mai stabil si mai bun
    for p in ema.parameters():
        p.requires_grad_(False)
    t.log(f"Retea: {model.blocks} blocuri x {model.channels} canale, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parametri")
    if n < 3_000_000 and model.blocks * model.channels >= 12 * 192:
        t.log("ATENTIE: sub ~3M pozitii o retea asa mare face overfitting repede; mai multe date sau o retea mai mica.")

    def ema_update(decay):
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

    batch, epochs, lr = CFG["batch"], CFG["epochs"], CFG["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = len(tr_idx) // batch
    if steps_per_epoch < 1:
        raise RuntimeError("Prea putine date pentru acest batch. Scade batch-ul sau pregateste mai multe pozitii.")
    total_steps = steps_per_epoch * epochs
    warmup = min(1000, max(1, total_steps // 10))

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        tt = (step - warmup) / max(1, total_steps - warmup)
        return lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * tt)))

    step, t0, ema_p, ema_v = 0, time.time(), None, None
    best, bad, saved = float("inf"), 0, False
    model.train()
    for epoch in range(1, epochs + 1):
        order = tr_idx[torch.randperm(len(tr_idx), device=tr_idx.device)]
        for i in range(steps_per_epoch):
            if t.stop:
                break
            b = order[i * batch:(i + 1) * batch]
            x, pi, pp, vb = data.get(b)
            xb = unpack(x).float()
            xb, pi = augment_flip(xb, pi)
            if cuda:
                xb = xb.contiguous(memory_format=torch.channels_last)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, v = model(xb)
                loss_p = policy_loss(logits, pi, pp)
                loss_v = F.mse_loss(v.float(), vb)
                loss = loss_p + loss_v
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            ema_update(min(0.999, (1 + step) / (10 + step)))
            if i % 50 == 0:
                lp, lv = loss_p.item(), loss_v.item()
                ema_p = lp if ema_p is None else 0.9 * ema_p + 0.1 * lp
                ema_v = lv if ema_v is None else 0.9 * ema_v + 0.1 * lv
                t.point("policy loss", ema_p)
                t.point("value loss", ema_v)
                t.progress = step / total_steps
                speed = step * batch / max(1e-6, time.time() - t0)
                t.status = f"epoca {epoch}/{epochs} | pas {i}/{steps_per_epoch} | policy {ema_p:.3f} | value {ema_v:.3f} | {speed:,.0f} poz/s"
        if t.stop:
            if saved:
                t.log("Oprit de utilizator. Ramane salvat cel mai bun model de pana acum.")
            elif had_model:
                t.log("Oprit inainte de prima epoca completa. Modelul vechi ramane neschimbat.")
            else:
                save_checkpoint(CHECKPOINT, ema, {"epoch": epoch})
                t.log("Oprit de utilizator. Model (partial) salvat.")
            return
        acc, vl_p, vl_v = evaluate(ema, data, val_idx, device, use_amp, amp_dtype)
        score = vl_p + vl_v
        if score < best:
            best, bad, saved = score, 0, True
            save_checkpoint(CHECKPOINT, ema, {"epoch": epoch, "val_score": score})
            tag = "NOU RECORD, salvat"
        else:
            bad += 1
            tag = f"fara progres ({bad}/{PATIENCE})"
        t.log(f"epoca {epoch}/{epochs}: top-1 = {acc * 100:.1f}% | policy {vl_p:.3f} | value MSE {vl_v:.3f} | {tag}")
        if bad >= PATIENCE:
            t.log("Validarea nu se mai imbunatateste -> opresc (evit overfitting). Modelul salvat e cel mai bun.")
            break
    t.progress = 1.0
    t.log("GATA! Compara cu modelul vechi in 'Duel' sau masoara ELO-ul in 'Arena'.")


# ======================================================================
# MCTS (PUCT, ca in AlphaZero) - cu cache, reutilizarea arborelui, fp16 si oprire timpurie
# ======================================================================
class Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior):
        self.prior, self.visits, self.value_sum, self.children = prior, 0, 0.0, {}

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    """MCTS PUCT cu evaluare in loturi si "virtual loss": GPU-ul evalueaza mai multe pozitii deodata.
    Cache-ul de evaluari (cheie = pozitia) evita sa calculeze de doua ori aceeasi pozitie."""
    def __init__(self, model, device, cpuct=1.5):
        self.device, self.cpuct = device, cpuct
        self.half = device.type == "cuda"
        self.model = model.eval()
        if self.half:
            self.model = self.model.half().to(memory_format=torch.channels_last)
        self.cache = {}
        self.cache_hits = self.net_evals = 0

    @torch.no_grad()
    def _net(self, planes_list):
        x = torch.from_numpy(np.stack(planes_list)).to(self.device)
        if self.half:
            x = x.half().contiguous(memory_format=torch.channels_last)
        else:
            x = x.float()
        logits, v = self.model(x)
        self.net_evals += len(planes_list)
        return logits.float().cpu().numpy(), v.float().cpu().numpy()

    @staticmethod
    def _priors(moves, row):
        idx = np.fromiter((i for _, i in moves), dtype=np.int64, count=len(moves))
        l = row[idx].astype(np.float64)
        l -= l.max()
        p = np.exp(l)
        p /= p.sum()
        return p.astype(np.float32)

    @staticmethod
    def _expand(node, moves, priors):
        for (m, _), p in zip(moves, priors):
            node.children[m] = Node(float(p))

    @staticmethod
    def _backup(path, v):
        for i in range(len(path) - 1, -1, -1):
            n = path[i]
            if i > 0:                      # scoate "virtual loss"-ul pus la selectie
                n.visits -= 1
                n.value_sum -= 1.0
            n.visits += 1
            n.value_sum += v
            v = -v

    def _select(self, node):
        sqrt_n = math.sqrt(node.visits + 1)
        fpu = node.q() - 0.15
        best, best_s = None, -1e18
        for m, c in node.children.items():
            q = -c.q() if c.visits else fpu
            s = q + self.cpuct * c.prior * sqrt_n / (1 + c.visits)
            if s > best_s:
                best, best_s = (m, c), s
        return best

    @staticmethod
    def _leaf_info(board):
        """(mutari, valoare terminala din perspectiva celui la mutare sau None)."""
        moves = legal_moves_with_index(board)
        if not moves:
            return moves, (-1.0 if board.is_check() else 0.0)
        if (board.is_insufficient_material() or board.halfmove_clock >= 100
                or (board.halfmove_clock >= 4 and board.is_repetition(2))):
            return moves, 0.0
        return moves, None

    def make_root(self, board):
        root = Node(0.0)
        moves = legal_moves_with_index(board)
        key = board._transposition_key()
        hit = self.cache.get(key)
        if hit is None:
            lg, vals = self._net([board_to_planes(board)])
            hit = (self._priors(moves, lg[0]), float(vals[0]))
            self.cache[key] = hit
        self._expand(root, moves, hit[0])
        return root

    @staticmethod
    def _decided(root, remaining):
        """True daca mutarea cea mai vizitata nu mai poate fi depasita cu simularile ramase."""
        if remaining <= 0:
            return True
        vs = sorted((c.visits for c in root.children.values()), reverse=True)
        return len(vs) < 2 or vs[0] - vs[1] > remaining

    def search(self, root, board, sims, batch=32, early=True):
        """Ruleaza `sims` simulari de la `root` (board e pozitia radacinii; e modificata si restaurata)."""
        batch = max(1, min(batch, sims // 8 or 1))
        done = 0
        while done < sims:
            n = min(batch, sims - done)
            pending = []
            for _ in range(n):
                node, path, pushed = root, [root], 0
                while node.children:
                    move, node = self._select(node)
                    board.push(move)
                    pushed += 1
                    path.append(node)
                    node.visits += 1                  # virtual loss
                    node.value_sum += 1.0
                moves, v = self._leaf_info(board)
                if v is None:
                    key = board._transposition_key()
                    hit = self.cache.get(key)
                    if hit is not None:
                        self.cache_hits += 1
                        self._expand(node, moves, hit[0])
                        v = hit[1]
                    else:
                        pending.append((path, board_to_planes(board), moves, key))
                for _ in range(pushed):
                    board.pop()
                if v is not None:
                    self._backup(path, v)
            done += n
            if pending:
                lg, vals = self._net([p[1] for p in pending])
                if len(self.cache) > CACHE_MAX:
                    self.cache.clear()
                for (path, _, moves, key), row, val in zip(pending, lg, vals):
                    pri, val = self._priors(moves, row), float(val)
                    self.cache[key] = (pri, val)
                    leaf = path[-1]
                    if not leaf.children:
                        self._expand(leaf, moves, pri)
                    self._backup(path, val)
            if early and self._decided(root, sims - done):
                break
        return done

    def think(self, root, board, sims):
        """Cauta `sims` simulari; daca primele doua mutari sunt apropiate, mai da 50% timp suplimentar."""
        done = self.search(root, board, sims)
        vs = sorted((c.visits for c in root.children.values()), reverse=True)
        if done >= sims and len(vs) >= 2 and vs[1] > 0.6 * vs[0]:
            done += self.search(root, board, max(1, sims // 2))
        return done


# ======================================================================
# TABLEBASE / MOTOARE (model propriu / Stockfish)
# ======================================================================
def tb_best_move(tb, board):
    """Cea mai buna mutare dupa tablebase Syzygy (castig cat mai rapid / infrangere cat mai lunga), sau None."""
    if chess.popcount(board.occupied) > 7 or board.castling_rights:
        return None
    best, best_key = None, None
    try:
        for m in list(board.legal_moves):
            board.push(m)
            try:
                wdl = -tb.probe_wdl(board)
                dtz = abs(tb.probe_dtz(board))
            finally:
                board.pop()
            key = (wdl, -dtz if wdl > 0 else (dtz if wdl < 0 else 0))
            if best_key is None or key > best_key:
                best, best_key = m, key
    except Exception:
        return None
    return best


class ModelEngine:
    def __init__(self, checkpoint, sims=400, name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_checkpoint(checkpoint, self.device)
        self.mcts = MCTS(model, self.device)
        self.sims = sims
        self.name = name or "AI-ul tau (MCTS)"
        self.last_info, self.last_value = "", None
        self.extras = True                          # carte de deschideri + tablebase (daca sunt setate in Setari)
        self.book = self.tb = None
        self._book_path = self._tb_path = ""
        self.prev = None                            # (stiva de mutari, mutarea jucata, radacina) pentru reutilizarea arborelui

    # ----- carte / tablebase (se (re)deschid cand se schimba calea in Setari) -----
    def _sync_extras(self):
        bp, sp = CFG.get("book", ""), CFG.get("syzygy", "")
        if bp != self._book_path:
            self._close_book()
            self._book_path = bp
            if bp and os.path.isfile(bp):
                try:
                    self.book = chess.polyglot.open_reader(bp)
                except Exception:
                    self.book = None
        if sp != self._tb_path:
            self._close_tb()
            self._tb_path = sp
            if sp and os.path.isdir(sp):
                try:
                    self.tb = chess.syzygy.open_tablebase(sp)
                except Exception:
                    self.tb = None

    def _close_book(self):
        if self.book is not None:
            try:
                self.book.close()
            except Exception:
                pass
        self.book = None

    def _close_tb(self):
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass
        self.tb = None

    def reset(self):
        self.prev = None

    def _reuse_root(self, board):
        """Daca adversarul a raspuns cu o mutare deja explorata, pastram subarborele (nu pornim de la zero)."""
        if not self.prev:
            return None
        stack, move, root = self.prev
        cur = board.move_stack
        k = len(stack)
        if len(cur) != k + 2 or cur[:k] != stack or cur[k] != move:
            return None
        child = root.children.get(move)
        reply = child.children.get(cur[-1]) if child else None
        if reply is None or not reply.children:
            return None
        return reply

    def best_move(self, board):
        t0 = time.time()
        self.last_value = None
        if self.extras and CFG.get("use_extras", True):
            self._sync_extras()
            if self.book is not None and board.ply() < 40:
                try:
                    m = self.book.weighted_choice(board).move
                    if m in board.legal_moves:
                        self.last_info, self.prev = "carte de deschideri", None
                        return m
                except IndexError:
                    pass
            if self.tb is not None:
                m = tb_best_move(self.tb, board)
                if m is not None:
                    self.last_info, self.prev = "tablebase Syzygy", None
                    return m
        for m in board.legal_moves:                  # mat in 1: nu mai cautam
            if board.gives_check(m):
                board.push(m)
                mate = board.is_checkmate()
                board.pop()
                if mate:
                    self.last_info, self.prev = "mat in 1", None
                    return m
        root = self._reuse_root(board)
        reused = root is not None
        if root is None:
            root = self.mcts.make_root(board)
        done = self.mcts.think(root, board.copy(), self.sims)
        move = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        self.last_value = root.q()                   # din perspectiva celui care a mutat
        self.prev = (list(board.move_stack), move, root)
        dt = max(1e-6, time.time() - t0)
        self.last_info = (f"{done} sim | {done / dt:,.0f} sim/s | eval {root.q():+.2f} | "
                          f"{root.children[move].visits} vizite{' | arbore refolosit' if reused else ''}")
        return move

    def close(self):
        self._close_book()
        self._close_tb()


class StockfishEngine:
    def __init__(self, path, elo=None, movetime=0.2):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.movetime = movetime
        self.last_info, self.last_value = "", None
        self.elo = None
        if elo:
            opt = self.engine.options["UCI_Elo"]
            elo = max(opt.min, min(opt.max, elo))
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            self.name = f"Stockfish (~{elo})"
            self.elo = elo
        else:
            self.name = "Stockfish (maxim)"

    def reset(self):
        pass

    def best_move(self, board):
        return self.engine.play(board, chess.engine.Limit(time=self.movetime)).move

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


# ---------------- arena ----------------
def perf_elo(score, n, opp_elo):
    s = min(max(score, 0.5 / n), 1 - 0.5 / n)
    return opp_elo + 400 * math.log10(s / (1 - s))


def score_ci(w, d, l):
    """scor + interval de incredere 95% (aproximare normala)."""
    n = w + d + l
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    se = math.sqrt(var / n)
    return s, max(0.0, s - 1.96 * se), min(1.0, s + 1.96 * se)


def random_opening(plies, rng):
    b = chess.Board()
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(rng.choice(moves))
    return list(b.move_stack)


def play_game(white, black, stop, opening=(), max_plies=300):
    board = chess.Board()
    for m in opening:
        board.push(m)
    white.reset()
    black.reset()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if stop():
            return "stop"
        eng = white if board.turn == chess.WHITE else black
        board.push(eng.best_move(board))
    out = board.outcome(claim_draw=True)
    return out.winner if out else None       # True=alb, False=negru, None=remiza


def run_match(t, a, b, games, label, rng):
    """a vs b, perechi de partide cu aceeasi deschidere si culori inversate. Returneaza (w, d, l) din punctul de vedere al lui a."""
    w = d = l = 0
    opening = []
    for g in range(games):
        if g % 2 == 0:
            opening = random_opening(ARENA_OPENING_PLIES, rng)
        a_white = g % 2 == 0
        res = play_game(a if a_white else b, b if a_white else a, lambda: t.stop, opening)
        if res == "stop":
            return None
        if res is None:
            d += 1
        elif res == a_white:
            w += 1
        else:
            l += 1
        t.status = f"{label}: joc {g + 1}/{games} | +{w} ={d} -{l}"
        t.progress += 1.0 / games / t.parts
    return w, d, l


def task_arena(t: Task):
    """Modelul vs Stockfish limitat la cateva nivele de ELO."""
    if not os.path.exists(CHECKPOINT):
        raise RuntimeError("Nu exista inca un model antrenat.")
    if not os.path.exists(CFG["stockfish"]):
        raise RuntimeError(f"Nu gasesc Stockfish la: {CFG['stockfish']} (alege-l din Setari)")
    bot = ModelEngine(CHECKPOINT, sims=CFG["sims"])
    games, levels = CFG["arena_games"], CFG["arena_elos"]
    rng = random.Random(1234)
    t.parts = len(levels)
    t.progress = 0.0
    estimates = []
    for elo in levels:
        sf = StockfishEngine(CFG["stockfish"], elo=elo, movetime=ARENA_MOVETIME)
        try:
            res = run_match(t, bot, sf, games, f"vs Stockfish {sf.elo or elo}", rng)
        finally:
            sf.close()
        if res is None:
            t.log("Oprit de utilizator.")
            return
        w, d, l = res
        s, lo, hi = score_ci(w, d, l)
        opp = sf.elo or elo
        est = perf_elo(s, games, opp)
        estimates.append((est, s))
        t.log(f"vs Stockfish {opp}: +{w} ={d} -{l} | scor {s:.2f} | ELO ~{est:.0f} (95%: {perf_elo(lo, games, opp):.0f} .. {perf_elo(hi, games, opp):.0f})")
    good = [e for e, s in estimates if 0.1 <= s <= 0.9] or [e for e, _ in estimates]
    t.progress = 1.0
    t.log(f"ELO estimat: ~{sum(good) / len(good):.0f}  (nivelele cu scor extrem, <10% sau >90%, nu conteaza cand exista altele)")


def task_duel(t: Task):
    """Modelul nou vs cel dinainte de ultima antrenare (checkpoints/previous.pt)."""
    if not os.path.exists(CHECKPOINT):
        raise RuntimeError("Nu exista inca un model antrenat.")
    if not os.path.exists(PREVIOUS):
        raise RuntimeError("Nu exista un model vechi. Dupa a doua antrenare, cel dinainte se salveaza automat ca previous.pt.")
    new = ModelEngine(CHECKPOINT, sims=CFG["sims"], name="nou")
    old = ModelEngine(PREVIOUS, sims=CFG["sims"], name="vechi")
    new.extras = old.extras = False                 # duel curat: fara carte / tablebase
    games = CFG["arena_games"]
    t.parts = 1
    t.progress = 0.0
    res = run_match(t, new, old, games, "nou vs vechi", random.Random(4321))
    if res is None:
        t.log("Oprit de utilizator.")
        return
    w, d, l = res
    s, lo, hi = score_ci(w, d, l)
    diff = perf_elo(s, games, 0)
    t.progress = 1.0
    t.log(f"nou vs vechi: +{w} ={d} -{l} | scor {s:.2f} (95%: {lo:.2f} .. {hi:.2f}) | diferenta ~{diff:+.0f} ELO")
    if lo > 0.5:
        t.log("VERDICT: modelul nou e clar mai bun.")
    elif s >= 0.55:
        t.log("VERDICT: probabil mai bun, dar intervalul inca include 50%. Mai multe partide ajuta.")
    elif hi < 0.5:
        t.log("VERDICT: modelul nou e mai slab. Pentru a reveni, copiaza checkpoints/previous.pt peste checkpoints/model.pt.")
    else:
        t.log("VERDICT: nu se vede o diferenta. Creste numarul de partide.")


# ---------------- reparare PyTorch CUDA ----------------
def launch_cuda_fix():
    """Scrie un .bat care (dupa ce aplicatia se inchide) reinstaleaza PyTorch cu CUDA."""
    idx = f"https://download.pytorch.org/whl/{CFG['cuda_tag']}"
    py = sys.executable
    lines = [
        "@echo off",
        "echo Astept inchiderea aplicatiei...",
        "timeout /t 3 /nobreak >nul",
        f'"{py}" -m pip uninstall -y torch',
        f'"{py}" -m pip install torch --index-url {idx}',
        f'"{py}" -c "import torch; print(\'CUDA disponibil:\', torch.cuda.is_available())"',
        "echo.",
        "echo Daca mai sus scrie True: porneste din nou chess_bot.py. Daca a dat eroare sau scrie False,",
        "echo incearca alta versiune CUDA din Setari, sau Python 3.12 (are cele mai multe pachete gata).",
        "pause",
    ]
    path = os.path.join(APP_DIR, "fix_cuda.bat")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\r\n".join(lines) + "\r\n")
    subprocess.Popen(["cmd", "/c", "start", "Fix CUDA", "cmd", "/k", path])


# ======================================================================
# INTERFATA GRAFICA
# ======================================================================
W, H = 1100, 720
C = dict(bg=(22, 24, 30), panel=(38, 41, 52), panel2=(54, 58, 74), text=(236, 238, 243), dim=(150, 156, 172),
         accent=(86, 156, 255), good=(88, 200, 128), bad=(232, 92, 92), warn=(238, 188, 78),
         light=(240, 217, 181), dark=(181, 136, 99), sel=(246, 246, 105), last=(205, 210, 106))
GLYPH = {chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝", chess.ROOK: "♜", chess.QUEEN: "♛", chess.KING: "♚"}
FONT = {}


def init_fonts():
    def sysf(names, size, bold=False):
        return pygame.font.SysFont(names, size, bold=bold)
    FONT["title"] = sysf("segoeui,arial,helvetica", 58, True)
    FONT["h1"] = sysf("segoeui,arial,helvetica", 32, True)
    FONT["h2"] = sysf("segoeui,arial,helvetica", 22, True)
    FONT["b"] = sysf("segoeui,arial,helvetica", 19)
    FONT["s"] = sysf("segoeui,arial,helvetica", 15)
    FONT["mono"] = sysf("consolas,couriernew,monospace", 14)


def pick_glyph_font(size):
    for name in ("segoeuisymbol", "dejavusans", "arialunicodems", "symbola", "freeserif", "applesymbols"):
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)


def lighten(c, n):
    return tuple(min(255, v + n) for v in c)


def draw_text(surf, s, font, color, pos, anchor="topleft"):
    img = font.render(s, True, color)
    r = img.get_rect()
    setattr(r, anchor, pos)
    surf.blit(img, r)
    return r


def wrap(font, s, width):
    words, lines, cur = s.split(" "), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_background():
    bg = pygame.Surface((W, H))
    for y in range(H):
        k = y / H
        pygame.draw.line(bg, (int(26 - 10 * k), int(29 - 11 * k), int(40 - 14 * k)), (0, y), (W, y))
    return bg


def draw_piece(surf, font, piece, center):
    g = GLYPH[piece.piece_type]
    fg, bg = ((250, 250, 250), (20, 20, 20)) if piece.color == chess.WHITE else ((28, 28, 28), (205, 205, 205))
    s_bg, s_fg = font.render(g, True, bg), font.render(g, True, fg)
    r = s_fg.get_rect(center=center)
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1)):
        surf.blit(s_bg, r.move(dx, dy))
    surf.blit(s_fg, r)


class Button:
    STYLES = {"primary": "accent", "ghost": "panel2", "danger": "bad", "good": "good"}

    def __init__(self, rect, label, cb, style="ghost", enabled=True):
        self.rect, self.label, self.cb, self.style, self.enabled = pygame.Rect(rect), label, cb, style, enabled

    def draw(self, surf):
        hover = self.enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        base = C[self.STYLES[self.style]]
        col = (62, 65, 76) if not self.enabled else (lighten(base, 22) if hover else base)
        pygame.draw.rect(surf, col, self.rect, border_radius=12)
        draw_text(surf, self.label, FONT["b"], C["text"] if self.enabled else C["dim"], self.rect.center, "center")

    def click(self, pos):
        if self.enabled and self.rect.collidepoint(pos):
            self.cb()
            return True
        return False


class Stepper:
    """Eticheta  [<]  valoare  [>]  - alege dintr-o lista de valori si salveaza in CFG."""
    def __init__(self, x, y, w, label, key, choices, fmt=str, on_change=None):
        self.x, self.y, self.w, self.label, self.key, self.choices, self.fmt, self.on_change = x, y, w, label, key, choices, fmt, on_change
        self.left = pygame.Rect(x + w - 190, y, 36, 34)
        self.right = pygame.Rect(x + w - 36, y, 36, 34)

    def index(self):
        try:
            return self.choices.index(CFG[self.key])
        except ValueError:
            return 0

    def draw(self, surf):
        draw_text(surf, self.label, FONT["b"], C["text"], (self.x, self.y + 17), "midleft")
        mouse = pygame.mouse.get_pos()
        for r, sym in ((self.left, "<"), (self.right, ">")):
            col = lighten(C["panel2"], 25) if r.collidepoint(mouse) else C["panel2"]
            pygame.draw.rect(surf, col, r, border_radius=8)
            draw_text(surf, sym, FONT["h2"], C["text"], r.center, "center")
        mid = (self.left.right + self.right.left) // 2
        val = CFG[self.key] if CFG[self.key] in self.choices else self.choices[0]
        draw_text(surf, self.fmt(val), FONT["b"], C["accent"], (mid, self.y + 17), "center")

    def click(self, pos):
        i = self.index()
        if self.left.collidepoint(pos):
            i = max(0, i - 1)
        elif self.right.collidepoint(pos):
            i = min(len(self.choices) - 1, i + 1)
        else:
            return False
        CFG[self.key] = self.choices[i]
        save_cfg()
        if self.on_change:
            self.on_change()
        return True


def draw_chart(surf, rect, series, empty_msg):
    pygame.draw.rect(surf, C["panel"], rect, border_radius=12)
    allv = [v for s in series.values() for v in s]
    if len(allv) < 2:
        draw_text(surf, empty_msg, FONT["s"], C["dim"], rect.center, "center")
        return
    lo, hi = min(allv), max(allv)
    if hi - lo < 1e-9:
        hi = lo + 1
    inner = pygame.Rect(rect.x + 14, rect.y + 34, rect.w - 28, rect.h - 48)
    colors = [C["accent"], C["warn"], C["good"]]
    for i, (name, s) in enumerate(series.items()):
        step = max(1, len(s) // 400)
        ss = s[::step]
        draw_text(surf, f"■ {name}: {s[-1]:.3f}", FONT["s"], colors[i % 3], (rect.x + 14 + i * 190, rect.y + 8))
        if len(ss) < 2:
            continue
        pts = [(inner.left + inner.w * k / (len(ss) - 1), inner.bottom - inner.h * (v - lo) / (hi - lo)) for k, v in enumerate(ss)]
        pygame.draw.lines(surf, colors[i % 3], False, pts, 2)


def pick_file(title, types):
    try:
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk()
        r.withdraw()
        r.attributes("-topmost", True)
        p = filedialog.askopenfilename(title=title, filetypes=types)
        r.destroy()
        return p or ""
    except Exception:
        return ""


def pick_dir(title):
    try:
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk()
        r.withdraw()
        r.attributes("-topmost", True)
        p = filedialog.askdirectory(title=title)
        r.destroy()
        return p or ""
    except Exception:
        return ""


# ---------------- ecrane ----------------
class Screen:
    def __init__(self, app):
        self.app = app

    def event(self, e): pass
    def update(self): pass
    def draw(self, surf): pass
    def leave(self): pass


class MenuScreen(Screen):
    def __init__(self, app):
        super().__init__(app)
        a = app
        items = [("Joaca vs AI-ul tau", "primary", lambda: a.go(GameScreen(a, False))),
                 ("Joaca vs Stockfish", "ghost", lambda: a.go(GameScreen(a, True))),
                 ("1.  Pregateste datele", "ghost", lambda: a.go(make_prepare_screen(a))),
                 ("2.  Antreneaza AI-ul", "ghost", lambda: a.go(make_train_screen(a))),
                 ("Arena: ELO vs Stockfish", "ghost", lambda: a.go(make_arena_screen(a))),
                 ("Duel: model nou vs vechi", "ghost", lambda: a.go(make_duel_screen(a))),
                 ("Setari", "ghost", lambda: a.go(SettingsScreen(a))),
                 ("Iesire", "danger", a.quit)]
        self.buttons = [Button((80, 200 + i * 56, 360, 48), lab, cb, st) for i, (lab, st, cb) in enumerate(items)]
        self.fix_btn = Button((634, 430, 382, 48), "Fix GPU: instaleaza PyTorch CUDA", self.do_fix, "good")
        self.glyph_font = pick_glyph_font(300)
        self.cache, self.cache_t = {}, 0

    def do_fix(self):
        if os.name != "nt":
            return
        launch_cuda_fix()
        self.app.quit()

    def status(self):
        if time.time() - self.cache_t > 1.0:
            cuda = torch.cuda.is_available()
            self.cache = dict(
                cuda=cuda, gpu=torch.cuda.get_device_name(0) if cuda else "",
                nvidia=shutil.which("nvidia-smi") is not None,
                model=os.path.exists(CHECKPOINT),
                shards=len(glob.glob(os.path.join(DATA_DIR, "shard_*.npz"))),
                sf=os.path.exists(CFG["stockfish"]))
            self.cache_t = time.time()
        return self.cache

    def event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for b in self.buttons:
                if b.click(e.pos):
                    return
            if not self.status()["cuda"]:
                self.fix_btn.click(e.pos)

    def draw(self, surf):
        # piesa decorativa
        deco = self.glyph_font.render(GLYPH[chess.KNIGHT], True, (255, 255, 255))
        deco.set_alpha(14)
        surf.blit(deco, (760, 330))
        draw_text(surf, "AI CHESS", FONT["title"], C["text"], (76, 66))
        draw_text(surf, "BOT", FONT["title"], C["accent"], (76 + FONT["title"].size("AI CHESS ")[0], 66))
        draw_text(surf, "antrenat de la zero pe placa ta video", FONT["b"], C["dim"], (80, 150))
        for b in self.buttons:
            b.draw(surf)

        st = self.status()
        panel = pygame.Rect(610, 70, 430, 430)
        pygame.draw.rect(surf, C["panel"], panel, border_radius=16)
        draw_text(surf, "Stare sistem", FONT["h2"], C["text"], (panel.x + 24, panel.y + 18))
        rows = [
            (st["cuda"], f"GPU: {st['gpu']}" if st["cuda"] else "GPU: PyTorch nu vede placa (varianta CPU)"),
            (st["shards"] > 0, f"Date: {st['shards']} shard-uri pregatite" if st["shards"] else "Date: lipsesc  ->  pasul 1"),
            (st["model"], "Model antrenat: gasit" if st["model"] else "Model antrenat: lipseste  ->  pasul 2"),
            (st["sf"], "Stockfish: gasit" if st["sf"] else "Stockfish: negasit (optional, vezi Setari)"),
        ]
        for i, (ok, label) in enumerate(rows):
            y = panel.y + 72 + i * 42
            pygame.draw.circle(surf, C["good"] if ok else (C["bad"] if i == 0 else C["warn"]), (panel.x + 34, y + 11), 8)
            draw_text(surf, label, FONT["b"], C["text"], (panel.x + 56, y))
        if not st["cuda"]:
            msg = ("Am detectat o placa NVIDIA, dar PyTorch e varianta CPU. Apasa butonul verde, asteapta instalarea, apoi porneste din nou aplicatia."
                   if st["nvidia"] else
                   "Nu am detectat o placa NVIDIA (nvidia-smi lipseste). Fara NVIDIA, antrenarea ruleaza pe CPU (lent). Daca ai NVIDIA, instaleaza driverul.")
            for i, ln in enumerate(wrap(FONT["s"], msg, panel.w - 48)):
                draw_text(surf, ln, FONT["s"], C["warn"], (panel.x + 24, panel.y + 250 + i * 20))
            if os.name == "nt":
                self.fix_btn.draw(surf)
        else:
            draw_text(surf, "Totul e pregatit pentru antrenare pe GPU.", FONT["b"], C["good"], (panel.x + 24, panel.y + 250))


class SettingsScreen(Screen):
    def __init__(self, app):
        super().__init__(app)
        n = lambda v: f"{v:,}"
        elo = lambda v: "Maxim" if v is None else str(v)
        yn = lambda v: "Da" if v else "Nu"
        src_fmt = lambda v: "Eval Lichess" if v == "evaldb" else "PGN+Stockfish"
        a, b, w = 60, 580, 460
        self.steppers = [
            # --- coloana stanga: date + antrenare ---
            Stepper(a, 120, w, "Sursa de date", "data_source", ["evaldb", "pgn"], src_fmt),
            Stepper(a, 160, w, "Nr. pozitii", "target_positions", [200_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000], n),
            Stepper(a, 200, w, "PGN: ELO minim", "min_elo", [1200, 1500, 1800, 2000, 2200, 2400]),
            Stepper(a, 240, w, "PGN: adancime SF", "sf_depth", [8, 10, 12, 14, 16]),
            Stepper(a, 280, w, "PGN: poz. / partida", "pos_per_game", [5, 10, 15, 20]),
            Stepper(a, 320, w, "Adauga la datele vechi", "append_data", [False, True], yn),
            Stepper(a, 388, w, "Epoci de antrenare", "epochs", [1, 2, 3, 5, 10, 15, 20, 30, 40]),
            Stepper(a, 428, w, "Batch", "batch", [128, 256, 512, 1024, 2048]),
            Stepper(a, 468, w, "Blocuri retea", "blocks", [4, 6, 8, 10, 12, 15, 20]),
            Stepper(a, 508, w, "Canale retea", "channels", [64, 96, 128, 192, 256]),
            Stepper(a, 548, w, "Continua modelul salvat", "resume", [False, True], lambda v: "Da" if v else "Nu"),
            # --- coloana dreapta: joc, arena, gpu ---
            Stepper(b, 120, w, "Simulari MCTS (putere)", "sims", [50, 100, 200, 400, 800, 1600, 3200]),
            Stepper(b, 160, w, "Joci cu", "play_white", [True, False], lambda v: "Albul" if v else "Negrul"),
            Stepper(b, 200, w, "ELO Stockfish", "stockfish_elo", [None, 1350, 1500, 1800, 2000, 2200, 2500, 2800], elo),
            Stepper(b, 240, w, "Carte + tablebase", "use_extras", [True, False], yn),
            Stepper(b, 306, w, "Partide / nivel / duel", "arena_games", [10, 20, 40, 100, 200]),
            Stepper(b, 346, w, "Nivele arena", "arena_elos",
                    [[1400, 1600, 1800], [1400], [1600], [1800], [2000], [2200], [1400, 1800, 2200]],
                    lambda v: "/".join(str(x) for x in v)),
            Stepper(b, 412, w, "Versiune CUDA (Fix GPU)", "cuda_tag", ["cu126", "cu128", "cu130"]),
        ]
        self.buttons = [
            Button((b, 480, 220, 40), "Fisier date local", self.pick_data),
            Button((b + 240, 480, 220, 40), "Date online (Lichess)", self.use_online),
            Button((b, 526, 220, 40), "Alege Stockfish", self.pick_sf),
            Button((b + 240, 526, 220, 40), "Carte deschideri", self.pick_book),
            Button((b, 572, 220, 40), "Folder Syzygy", self.pick_syzygy),
            Button((880, 650, 160, 46), "Inapoi", lambda: app.go(MenuScreen(app)), "primary"),
        ]

    def pick_data(self):
        p = pick_file("Alege fisierul de date (.jsonl / .pgn / .zst)", [("Date", "*.jsonl *.pgn *.zst"), ("Toate", "*.*")])
        if p:
            CFG["eval_src" if CFG["data_source"] == "evaldb" else "pgn"] = p
            save_cfg()

    def use_online(self):
        if CFG["data_source"] == "evaldb":
            CFG["eval_src"] = DEFAULT_EVAL_URL
        else:
            CFG["pgn"] = DEFAULT_PGN_URL
        save_cfg()

    def pick_sf(self):
        p = pick_file("Alege stockfish.exe", [("Executabil", "*.exe"), ("Toate", "*.*")])
        if p:
            CFG["stockfish"] = p
            save_cfg()

    def pick_book(self):
        p = pick_file("Alege cartea de deschideri Polyglot (.bin)", [("Polyglot", "*.bin"), ("Toate", "*.*")])
        if p:
            CFG["book"] = p
            save_cfg()

    def pick_syzygy(self):
        p = pick_dir("Alege folderul cu tablebase Syzygy")
        if p:
            CFG["syzygy"] = p
            save_cfg()

    def event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for s in self.steppers:
                if s.click(e.pos):
                    return
            for b in self.buttons:
                if b.click(e.pos):
                    return

    def draw(self, surf):
        draw_text(surf, "Setari", FONT["h1"], C["text"], (60, 36))
        for title, x, y in (("Date", 60, 98), ("Antrenare", 60, 366), ("Joc", 580, 98), ("Arena / duel", 580, 284),
                            ("Instalare GPU", 580, 390), ("Fisiere", 580, 458)):
            draw_text(surf, title.upper(), FONT["s"], C["accent"], (x, y))
        for s in self.steppers:
            s.draw(surf)
        online = CFG["data_source"] == "evaldb"
        src = CFG["eval_src"] if online else CFG["pgn"]
        online_default = DEFAULT_EVAL_URL if online else DEFAULT_PGN_URL
        shown = "Lichess online (streaming)" if src == online_default else src
        lines = [
            ("Date: " + shown[-68:], C["dim"]),
            ("Stockfish: " + CFG["stockfish"][-62:], C["good"] if os.path.exists(CFG["stockfish"]) else C["warn"]),
            ("Carte: " + (CFG["book"][-64:] if CFG["book"] else "(nesetata)"), C["good"] if CFG["book"] and os.path.isfile(CFG["book"]) else C["dim"]),
            ("Syzygy: " + (CFG["syzygy"][-62:] if CFG["syzygy"] else "(nesetat)"), C["good"] if CFG["syzygy"] and os.path.isdir(CFG["syzygy"]) else C["dim"]),
        ]
        for i, (txt, col) in enumerate(lines):
            draw_text(surf, txt, FONT["s"], col, (60, 600 + i * 20))
        for b in self.buttons:
            b.draw(surf)


class TaskScreen(Screen):
    def __init__(self, app, title, subtitle, fn, chart=False, pre_check=None):
        super().__init__(app)
        self.title, self.subtitle, self.fn, self.chart, self.pre_check = title, subtitle, fn, chart, pre_check
        self.task = None
        self.btn_main = Button((60, 650, 220, 46), "Start", self.on_main, "good")
        self.btn_back = Button((880, 650, 160, 46), "Inapoi", lambda: app.go(MenuScreen(app)), "ghost")
        self.warn = ""
        self.warn_t = 0

    def on_main(self):
        if self.task and not self.task.done:
            self.task.stop = True
            return
        self.task = Task(self.fn, self.title)
        self.task.start()

    def leave(self):
        if self.task and not self.task.done:
            self.task.stop = True

    def event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.btn_main.click(e.pos) or self.btn_back.click(e.pos)

    def draw(self, surf):
        t = self.task
        running = t is not None and not t.done
        self.btn_main.label = "Stop" if running else ("Porneste din nou" if t else "Start")
        self.btn_main.style = "danger" if running else "good"
        if time.time() - self.warn_t > 1.0:
            self.warn = self.pre_check() if self.pre_check else ""
            self.warn_t = time.time()

        draw_text(surf, self.title, FONT["h1"], C["text"], (60, 30))
        for i, ln in enumerate(wrap(FONT["s"], self.subtitle, 980)):
            draw_text(surf, ln, FONT["s"], C["dim"], (60, 78 + i * 18))
        y = 120
        if self.warn and not running:
            for i, ln in enumerate(wrap(FONT["s"], self.warn, 980)):
                draw_text(surf, ln, FONT["s"], C["warn"], (60, y + i * 18))
            y += 18 * len(wrap(FONT["s"], self.warn, 980)) + 6
        if self.chart:
            crect = pygame.Rect(60, y, 980, 200)
            draw_chart(surf, crect, t.series if t else {}, "graficul loss-ului apare dupa ce porneste antrenarea")
            y += 212
        log_rect = pygame.Rect(60, y, 980, 590 - y)
        pygame.draw.rect(surf, (16, 17, 22), log_rect, border_radius=12)
        lines = (t.lines if t else [])
        rows = (log_rect.h - 16) // 18
        for i, ln in enumerate(lines[-rows:]):
            col = C["bad"] if ln.startswith("EROARE") else (C["warn"] if ln.startswith("ATENTIE") else C["text"])
            draw_text(surf, ln[:135], FONT["mono"], col, (log_rect.x + 12, log_rect.y + 8 + i * 18))
        # bara de progres
        bar = pygame.Rect(60, 604, 980, 14)
        pygame.draw.rect(surf, C["panel"], bar, border_radius=7)
        if t and t.progress > 0:
            fill = bar.copy()
            fill.w = max(14, int(bar.w * min(1.0, t.progress)))
            pygame.draw.rect(surf, C["bad"] if t.failed else C["accent"], fill, border_radius=7)
        status = t.status if t else ""
        if t and t.done:
            status = "EROARE" if t.failed else "Terminat"
        draw_text(surf, status, FONT["s"], C["dim"], (60, 624))
        self.btn_main.draw(surf)
        self.btn_back.draw(surf)


def make_prepare_screen(app):
    def pre():
        if CFG["data_source"] == "evaldb":
            src = CFG["eval_src"]
            if not src.startswith("http") and not os.path.exists(src):
                return f"Fisierul nu exista: {src}. Alege altul din Setari sau foloseste datele online."
            if src.startswith("http"):
                return "Se citesc evaluari Stockfish deja calculate, direct de pe Lichess (internet necesar, nu descarci tot fisierul). Nu ai nevoie de Stockfish."
            return ""
        src = CFG["pgn"]
        if not src.startswith("http") and not os.path.exists(src):
            return f"Fisierul nu exista: {src}. Alege altul din Setari sau foloseste datele online."
        if not os.path.exists(CFG["stockfish"]):
            return "Stockfish lipseste: fara el etichetele sunt mutarile oamenilor (mult mai slabe). Alege stockfish.exe in Setari sau treci pe 'Eval Lichess'."
        return ""
    return TaskScreen(app, "Pregateste datele",
                      "Construieste pozitiile de antrenare (mutari bune + evaluare). Sursa si numarul de pozitii se aleg in Setari: cu cat mai multe, cu atat mai bine (tinta: 5-20 de milioane).",
                      task_prepare, pre_check=pre)


def make_train_screen(app):
    def pre():
        if not glob.glob(os.path.join(DATA_DIR, "shard_*.npz")):
            return "Nu ai date de antrenare. Ruleaza intai pasul 1 (Pregateste datele)."
        if not torch.cuda.is_available():
            return "PyTorch NU vede GPU-ul: antrenarea va merge pe CPU, extrem de lent. Din meniu apasa 'Fix GPU'."
        return ""
    return TaskScreen(app, "Antreneaza AI-ul",
                      "Antrenare supervizata pe GPU (policy + value), validare pe partide separate. Poti apasa Stop oricand. Modelul vechi se pastreaza ca previous.pt pentru duel.",
                      task_train, chart=True, pre_check=pre)


def make_arena_screen(app):
    def pre():
        if not os.path.exists(CHECKPOINT):
            return "Nu exista inca un model antrenat."
        if not os.path.exists(CFG["stockfish"]):
            return "Nu gasesc Stockfish. Descarca-l de pe stockfishchess.org si alege stockfish.exe din Setari."
        return ""
    return TaskScreen(app, "Arena: masoara ELO-ul",
                      "Modelul joaca contra Stockfish limitat la nivelele alese in Setari, cu deschideri aleatorii pe perechi de partide. Cu 100+ partide pe nivel eroarea scade serios.",
                      task_arena, pre_check=pre)


def make_duel_screen(app):
    def pre():
        if not os.path.exists(CHECKPOINT):
            return "Nu exista inca un model antrenat."
        if not os.path.exists(PREVIOUS):
            return "Nu exista inca un model vechi (apare dupa a doua antrenare)."
        return ""
    return TaskScreen(app, "Duel: model nou vs model vechi",
                      "Cea mai buna metoda sa vezi daca o schimbare chiar ajuta. Fara carte / tablebase, deschideri aleatorii, cu interval de incredere.",
                      task_duel, pre_check=pre)


class GameScreen(Screen):
    SQ = 80
    BX, BY = 76, 40

    def __init__(self, app, use_stockfish):
        super().__init__(app)
        self.engine, self.error = None, None
        self.thinking, self.pending, self.game_id = False, None, 0
        try:
            if use_stockfish:
                if not os.path.exists(CFG["stockfish"]):
                    raise RuntimeError("Nu gasesc Stockfish. Descarca-l de pe stockfishchess.org si alege stockfish.exe din Setari.")
                self.engine = StockfishEngine(CFG["stockfish"], elo=CFG["stockfish_elo"])
            else:
                if not os.path.exists(CHECKPOINT):
                    raise RuntimeError("Nu ai inca un AI antrenat. Fa pasii 1 (date) si 2 (antrenare) din meniu.")
                self.engine = ModelEngine(CHECKPOINT, sims=CFG["sims"])
        except Exception as ex:
            self.error = str(ex)
        self.use_sf = use_stockfish
        self.human = chess.WHITE if CFG["play_white"] else chess.BLACK
        self.font_piece = pick_glyph_font(int(self.SQ * 0.82))
        self.new_game()
        px = 750
        self.btn_new = Button((px, 600, 144, 44), "Joc nou", self.new_game, "primary")
        self.btn_undo = Button((px + 156, 600, 144, 44), "Undo", self.undo)
        self.btn_flip = Button((px, 652 - 4, 144, 40), "Roteste", self.flip)
        self.btn_menu = Button((px + 156, 652 - 4, 144, 40), "Meniu", lambda: app.go(MenuScreen(app)))
        self.btn_back = Button((W // 2 - 80, 400, 160, 46), "Inapoi", lambda: app.go(MenuScreen(app)), "primary")
        self.sims_stepper = Stepper(750, 540, 300, "Putere", "sims", [50, 100, 200, 400, 800, 1600, 3200]) if not use_stockfish else None

    def leave(self):
        self.game_id += 1
        if self.engine and not self.thinking:
            self.engine.close()

    # ---------- stare ----------
    def new_game(self):
        self.game_id += 1
        self.board = chess.Board()
        self.sel, self.last, self.anim, self.pending = None, None, None, None
        self.sans, self.eval, self.eval_disp, self.msg = [], None, 0.5, ""
        if self.engine:
            self.engine.reset()

    def flip(self):
        self.human = not self.human
        self.new_game()

    def undo(self):
        if self.thinking or not self.board.move_stack:
            return
        for _ in range(2 if self.board.turn == self.human else 1):
            if self.board.move_stack:
                self.board.pop()
        self.last = self.board.peek() if self.board.move_stack else None
        self.sel, self.anim = None, None
        self.refresh_sans()

    def refresh_sans(self):
        b, out = chess.Board(), []
        for m in self.board.move_stack:
            out.append(b.san(m))
            b.push(m)
        self.sans = out

    def over(self):
        return self.board.outcome(claim_draw=True) is not None

    def flipped(self):
        return self.human == chess.BLACK

    def sq_xy(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped():
            return self.BX + (7 - f) * self.SQ, self.BY + r * self.SQ
        return self.BX + f * self.SQ, self.BY + (7 - r) * self.SQ

    def xy_sq(self, x, y):
        x, y = x - self.BX, y - self.BY
        if not (0 <= x < 8 * self.SQ and 0 <= y < 8 * self.SQ):
            return None
        f, r = x // self.SQ, y // self.SQ
        return chess.square(7 - f, r) if self.flipped() else chess.square(f, 7 - r)

    def play_move(self, move):
        piece = self.board.piece_at(move.from_square)
        self.anim = dict(piece=piece, frm=move.from_square, to=move.to_square, t0=time.time())
        self.board.push(move)
        self.last, self.sel = move, None
        self.refresh_sans()

    # ---------- bot ----------
    def start_bot(self):
        if isinstance(self.engine, ModelEngine):
            self.engine.sims = CFG["sims"]
        gid, snap = self.game_id, self.board.copy()
        mover = snap.turn
        self.thinking = True

        def work():
            move = None
            try:
                move = self.engine.best_move(snap)
            except Exception as ex:
                self.msg = f"Eroare motor: {ex}"
            if gid == self.game_id and move is not None:
                self.pending = (move, getattr(self.engine, "last_value", None), mover)
            self.thinking = False

        threading.Thread(target=work, daemon=True).start()

    def update(self):
        if self.error or not self.engine:
            return
        if self.pending:
            move, val, mover = self.pending
            self.pending = None
            if move in self.board.legal_moves:
                self.play_move(move)
                if val is not None:
                    self.eval = val if mover == chess.WHITE else -val
        if (not self.thinking and not self.pending and not self.over() and self.board.turn != self.human):
            self.start_bot()
        target = 0.5 if self.eval is None else (self.eval + 1) / 2
        self.eval_disp += (target - self.eval_disp) * 0.12

    # ---------- input ----------
    def event(self, e):
        if e.type != pygame.MOUSEBUTTONDOWN or e.button != 1:
            return
        if self.error:
            self.btn_back.click(e.pos)
            return
        for b in (self.btn_new, self.btn_undo, self.btn_flip, self.btn_menu):
            if b.click(e.pos):
                return
        if self.sims_stepper and self.sims_stepper.click(e.pos):
            return
        if self.board.turn != self.human or self.over():
            return
        sq = self.xy_sq(*e.pos)
        if sq is None:
            return
        if self.sel is not None:
            move = chess.Move(self.sel, sq)
            p = self.board.piece_at(self.sel)
            if p and p.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                move.promotion = chess.QUEEN
            if move in self.board.legal_moves:
                self.play_move(move)
                return
        piece = self.board.piece_at(sq)
        self.sel = sq if piece and piece.color == self.human else None

    # ---------- desen ----------
    def draw(self, surf):
        if self.error:
            pygame.draw.rect(surf, C["panel"], (W // 2 - 340, 220, 680, 230), border_radius=16)
            draw_text(surf, "Nu pot porni jocul", FONT["h1"], C["bad"], (W // 2, 255), "midtop")
            for i, ln in enumerate(wrap(FONT["b"], self.error, 620)):
                draw_text(surf, ln, FONT["b"], C["text"], (W // 2, 310 + i * 26), "midtop")
            self.btn_back.draw(surf)
            return
        S, BX, BY = self.SQ, self.BX, self.BY
        # bara de evaluare
        bar = pygame.Rect(BX - 32, BY, 18, 8 * S)
        pygame.draw.rect(surf, (30, 30, 34), bar, border_radius=6)
        wh = int(bar.h * self.eval_disp)
        wr = pygame.Rect(bar.x, bar.y if self.flipped() else bar.bottom - wh, bar.w, wh)
        pygame.draw.rect(surf, (240, 240, 240), wr, border_radius=6)
        # casute
        for sq in chess.SQUARES:
            x, y = self.sq_xy(sq)
            light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            col = C["light"] if light else C["dark"]
            if self.last and sq in (self.last.from_square, self.last.to_square):
                col = C["last"]
            if sq == self.sel:
                col = C["sel"]
            pygame.draw.rect(surf, col, (x, y, S, S))
        bottom_rank = 7 if self.flipped() else 0               # coordonate pe margini
        left_file = 7 if self.flipped() else 0
        for k in range(8):
            x, y = self.sq_xy(chess.square(k, bottom_rank))
            lt = (k + bottom_rank) % 2 == 1
            draw_text(surf, "abcdefgh"[k], FONT["s"], C["dark"] if lt else C["light"], (x + S - 13, y + S - 20))
            x, y = self.sq_xy(chess.square(left_file, k))
            lt = (left_file + k) % 2 == 1
            draw_text(surf, str(k + 1), FONT["s"], C["dark"] if lt else C["light"], (x + 4, y + 3))
        if self.sel is not None:                                # mutari posibile
            for m in self.board.legal_moves:
                if m.from_square == self.sel:
                    x, y = self.sq_xy(m.to_square)
                    cap = self.board.piece_at(m.to_square) is not None
                    if cap:
                        pygame.draw.circle(surf, (70, 70, 70), (x + S // 2, y + S // 2), S // 2 - 4, 4)
                    else:
                        pygame.draw.circle(surf, (70, 70, 70), (x + S // 2, y + S // 2), 10)
        # piese (+ animatie)
        anim = self.anim
        if anim and time.time() - anim["t0"] > 0.18:
            self.anim = anim = None
        for sq, piece in self.board.piece_map().items():
            if anim and sq == anim["to"]:
                continue
            x, y = self.sq_xy(sq)
            draw_piece(surf, self.font_piece, piece, (x + S // 2, y + S // 2))
        if anim:
            k = min(1.0, (time.time() - anim["t0"]) / 0.18)
            fx, fy = self.sq_xy(anim["frm"])
            tx, ty = self.sq_xy(anim["to"])
            draw_piece(surf, self.font_piece, anim["piece"], (fx + (tx - fx) * k + S // 2, fy + (ty - fy) * k + S // 2))
        # sfarsit de joc
        out = self.board.outcome(claim_draw=True)
        if out:
            ov = pygame.Surface((8 * S, 120), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 190))
            surf.blit(ov, (BX, BY + 4 * S - 60))
            if out.winner is None:
                txt, col = "Remiza", C["warn"]
            elif out.winner == self.human:
                txt, col = "Ai castigat!", C["good"]
            else:
                txt, col = "Ai pierdut", C["bad"]
            draw_text(surf, txt, FONT["h1"], col, (BX + 4 * S, BY + 4 * S - 22), "center")
            draw_text(surf, "apasa 'Joc nou' ca sa joci din nou", FONT["s"], C["text"], (BX + 4 * S, BY + 4 * S + 22), "center")
        # panou dreapta
        panel = pygame.Rect(730, 40, 340, 640)
        pygame.draw.rect(surf, C["panel"], panel, border_radius=16)
        draw_text(surf, self.engine.name, FONT["h2"], C["text"], (750, 58))
        if out:
            status = "Joc terminat"
        elif self.thinking:
            dots = "." * (int(time.time() * 3) % 4)
            status = "Se gandeste" + dots
        else:
            status = "Randul tau" + ("  (sah!)" if self.board.is_check() else "")
        draw_text(surf, status, FONT["b"], C["accent"], (750, 92))
        info = self.engine.last_info or "Alege o piesa, apoi casuta destinatie."
        for i, ln in enumerate(wrap(FONT["s"], self.msg or info, 300)[:2]):
            draw_text(surf, ln, FONT["s"], C["bad"] if self.msg else C["dim"], (750, 122 + i * 18))
        # lista de mutari
        pygame.draw.rect(surf, (28, 30, 38), (750, 168, 300, 300), border_radius=10)
        pairs = [(i // 2 + 1, self.sans[i], self.sans[i + 1] if i + 1 < len(self.sans) else "") for i in range(0, len(self.sans), 2)]
        for j, (num, w, b) in enumerate(pairs[-12:]):
            y = 178 + j * 24
            draw_text(surf, f"{num}.", FONT["s"], C["dim"], (764, y))
            draw_text(surf, w, FONT["b"], C["text"], (806, y - 3))
            draw_text(surf, b, FONT["b"], C["text"], (928, y - 3))
        if self.sims_stepper:
            self.sims_stepper.draw(surf)
        else:
            draw_text(surf, "Nivelul Stockfish se schimba din Setari", FONT["s"], C["dim"], (750, 548))
        for b in (self.btn_new, self.btn_undo, self.btn_flip, self.btn_menu):
            b.draw(surf)


class App:
    def __init__(self):
        pygame.init()
        self.surf = pygame.display.set_mode((W, H))
        pygame.display.set_caption("AI Chess Bot")
        init_fonts()
        self.bg = make_background()
        self.running = True
        self.screen = MenuScreen(self)

    def go(self, screen):
        self.screen.leave()
        self.screen = screen

    def quit(self):
        self.running = False

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                else:
                    self.screen.event(e)
            self.screen.update()
            self.surf.blit(self.bg, (0, 0))
            self.screen.draw(self.surf)
            pygame.display.flip()
            clock.tick(60)
        self.screen.leave()
        pygame.quit()


if __name__ == "__main__":
    if pygame is None:
        print("Lipseste pygame. Ruleaza:  pip install pygame-ce")
        sys.exit(1)
    load_cfg()
    App().run()