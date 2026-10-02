import json
import os
import random
import shutil
import sys
import urllib.request
import zipfile

import chess.engine
import Chess_bot as cb

HERE = os.path.dirname(os.path.abspath(__file__))
SF_DIR = os.path.join(HERE, "stockfish_bin")
RELEASE_API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
PREFER = ["universal", "avx2", "sse41-popcnt", "ssse3", "x86-64.zip"]  # ordinea preferintei

SIMS     = [50, 100, 200, 400, 800]    # valorile de simulari MCTS de testat
LEVELS   = [1350, 1500, 1700]          # nivele Stockfish (UCI_Elo)
GAMES    = 20                          # numar PAR (perechi cu culori inversate)
MOVETIME = 0.1                         # secunde / mutare pentru Stockfish


# ---------------- Stockfish: cautare / verificare / descarcare ----------------
def find_exe(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().startswith("stockfish") and f.lower().endswith(".exe"):
                return os.path.join(root, f)
    return None


def works(path):
    """True daca executabilul porneste si raspunde la protocolul UCI."""
    try:
        eng = chess.engine.SimpleEngine.popen_uci(path)
        eng.quit()
        return True
    except Exception:
        return False


def download_stockfish():
    print("Descarc Stockfish de pe GitHub...", flush=True)
    req = urllib.request.Request(RELEASE_API, headers={"User-Agent": "chess-bot"})
    rel = json.load(urllib.request.urlopen(req, timeout=30))
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    print("Versiune:", rel.get("tag_name"), flush=True)

    def rank(name):
        for i, p in enumerate(PREFER):
            if p in name:
                return i
        return len(PREFER)

    # arhive Windows x86-64 (fara ARM), cele mai compatibile/rapide primele
    zips = sorted((n for n in assets if n.endswith(".zip") and "windows" in n and "arm" not in n), key=rank)
    if not zips:
        print("Nu am gasit nicio arhiva Windows in release. Assets:", list(assets)[:10], flush=True)
        return None

    for name in zips:
        print(f"  descarc {name} (poate dura, ~80 MB) ...", flush=True)
        zpath = os.path.join(HERE, name)
        urllib.request.urlretrieve(assets[name], zpath)
        shutil.rmtree(SF_DIR, ignore_errors=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(SF_DIR)
        os.remove(zpath)
        exe = find_exe(SF_DIR)
        if exe and works(exe):
            return exe
        print("  nu porneste, incerc urmatoarea varianta", flush=True)
    return None


def ensure_stockfish():
    candidates = [cb.CFG.get("stockfish", ""), find_exe(SF_DIR) if os.path.isdir(SF_DIR) else None,
                  shutil.which("stockfish")]
    for c in candidates:
        if c and os.path.isfile(c) and works(c):
            print("Stockfish gasit:", c, flush=True)
            return c
    try:
        exe = download_stockfish()
    except Exception as ex:
        exe = None
        print("Descarcarea a esuat:", ex, flush=True)
    if not exe:
        sys.exit("Nu am reusit sa obtin Stockfish. Descarca-l manual de pe stockfishchess.org, "
                 "dezarhiveaza-l in C:\\stockfish\\ si alege .exe-ul in Setari (butonul 'Alege Stockfish').")
    print("Stockfish gata:", exe, flush=True)
    return exe


# ---------------- meciuri ----------------
def new_task():
    t = cb.Task(lambda t: None, "elo")
    t.parts, t.progress = 1, 0.0
    return t


def main():
    cb.load_cfg()
    if not os.path.exists(cb.CHECKPOINT):
        sys.exit(f"Nu gasesc modelul: {os.path.abspath(cb.CHECKPOINT)}. Antreneaza-l intai.")
    if GAMES % 2:
        sys.exit("GAMES trebuie sa fie par (perechi cu culori inversate).")

    sf_path = ensure_stockfish()
    cb.CFG["stockfish"] = sf_path          # il tine minte si aplicatia
    cb.save_cfg()

    rng = random.Random(1234)
    summary = []
    for sims in SIMS:
        bot = cb.ModelEngine(cb.CHECKPOINT, sims=sims)
        bot.extras = False                 # fara carte / tablebase: masuram doar reteaua + MCTS
        ests = []
        for lvl in LEVELS:
            sf = cb.StockfishEngine(sf_path, elo=lvl, movetime=MOVETIME)
            try:
                res = cb.run_match(new_task(), bot, sf, GAMES, f"{sims} sims vs {lvl}", rng)
            finally:
                sf.close()
            w, d, l = res
            opp = sf.elo or lvl
            s, lo, hi = cb.score_ci(w, d, l)
            est = cb.perf_elo(s, GAMES, opp)
            ests.append((est, s))
            print(f"sims={sims:4d} vs SF {opp}: +{w} ={d} -{l} | scor {s:.2f} | ELO ~{est:.0f}", flush=True)
        good = [e for e, s in ests if 0.1 <= s <= 0.9] or [e for e, _ in ests]
        summary.append((sims, sum(good) / len(good)))
        bot.close()

    print("\n=== ELO estimat in functie de simulari ===")
    for sims, elo in summary:
        print(f"{sims:5d} simulari -> ~{elo:.0f}")


if __name__ == "__main__":
    main()