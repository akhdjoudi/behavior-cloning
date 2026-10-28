"""Évaluation Stockfish pour la barre de l'appli (et plus tard pour style_bench.py).

Responsable : rôle 2. Voir la section correspondante du plan de projet.

Un seul processus Stockfish, partagé entre les requêtes (verrou), avec un cache par position.
Score toujours du point de vue des Blancs : centipions, ou mat en N coups (négatif si les Noirs matent).
"""

import shutil
import threading
from collections import OrderedDict
from pathlib import Path

import chess
import chess.engine

DEFAULT_PATHS = ("/usr/games/stockfish", "/usr/bin/stockfish", "/opt/homebrew/bin/stockfish")


def find_stockfish() -> str | None:
    found = shutil.which("stockfish")
    return found or next((p for p in DEFAULT_PATHS if Path(p).exists()), None)


class Evaluator:
    def __init__(self, path: str, time_limit: float = 0.35, depth: int = 20, threads: int = 2, cache_size: int = 2048):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": threads, "Hash": 64})
        self.name = self.engine.id.get("name", "Stockfish")
        self.limit = chess.engine.Limit(time=time_limit, depth=depth)
        self.lock = threading.Lock()
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.cache_size = cache_size

    def evaluate(self, fen: str) -> dict:
        board = chess.Board(fen)
        key = " ".join(fen.split()[:4])  # les compteurs de coups ne changent pas l'évaluation
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        if board.is_checkmate():
            result = {"cp": None, "mate": 0, "winner": "black" if board.turn == chess.WHITE else "white"}
        elif board.is_game_over():
            result = {"cp": 0, "mate": None}
        else:
            with self.lock:
                info = self.engine.analyse(board, self.limit)
            score = info["score"].white()
            pv = info.get("pv") or []
            result = {
                "cp": score.score(),
                "mate": score.mate(),
                "depth": info.get("depth"),
                "best": pv[0].uci() if pv else None,
                "best_san": board.san(pv[0]) if pv else None,
            }

        self.cache[key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def close(self) -> None:
        self.engine.quit()
