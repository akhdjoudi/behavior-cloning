"""Métriques de style via Stockfish et classifieur de style.

Responsable : rôle 2. Voir la section correspondante du plan de projet.

Pour chaque joueur, on tire au hasard des positions de son split test (hors ouverture) et on compare trois coups :
- `reel` : le coup que le joueur a joué ;
- `modele` : le coup préféré du modèle conditionné sur ce joueur ;
- `generique` : le coup préféré du même modèle avec l'identifiant générique.

Pour chaque source, on mesure la part de captures, d'échecs, de coups de pion, de dame et de roi, et (avec
Stockfish) la perte moyenne en centipions par rapport au meilleur coup. Le style est appris si, pour un joueur,
le profil de `modele` est plus proche de celui de `reel` que celui de `generique` (distance L1 entre les parts).
`contraste` donne l'écart moyen entre les profils réels de deux joueurs : si il est petit, les joueurs ne se
distinguent pas par ces mesures et le test ne prouve rien.

Avec 300 positions par joueur, une part se mesure à ± 3 points près environ.

    python -m chessstyle.style_bench runs/latest data/processed/icons/test.h5 --out runs/latest/style.json
"""

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path

import chess
import chess.engine
import h5py
import numpy as np
import torch

from chessstyle.encoding import PLANE_CASTLING, PLANE_EN_PASSANT, PLANE_HALFMOVE, ChessGame
from chessstyle.evaluate import OPENING_PLIES, load_model
from chessstyle.players import GENERIC, PLAYERS

FEATURES = ("capture", "echec", "pion", "dame", "roi")
SOURCES = ("reel", "modele", "generique")
MATE = 10_000
CPL_CAP = 1_000  # un coup qui perd la partie compte 1 000 centipions, pas plus, pour qu'un seul gros coup ne domine pas la moyenne


def board_from_planes(planes: np.ndarray) -> chess.Board:
    """Position canonique (trait aux Blancs) d'un tenseur d'entrée, pour retrouver pièces, échecs et captures."""
    board = chess.Board(None)
    for i in range(12):
        color = chess.WHITE if i < 6 else chess.BLACK
        for rank, file in zip(*np.nonzero(planes[i])):
            board.set_piece_at(chess.square(int(file), int(rank)), chess.Piece(i % 6 + 1, color))
    board.turn = chess.WHITE
    board.set_castling_fen("".join(c for i, c in enumerate("KQkq") if planes[PLANE_CASTLING + i].any()) or "-")
    if planes[PLANE_EN_PASSANT].any():
        rank, file = np.argwhere(planes[PLANE_EN_PASSANT])[0]
        board.ep_square = chess.square(int(file), int(rank))
    board.halfmove_clock = int(planes[PLANE_HALFMOVE, 0, 0])
    return board


def features(board: chess.Board, move: chess.Move) -> list[bool]:
    piece = board.piece_type_at(move.from_square)
    return [board.is_capture(move), board.gives_check(move), piece == chess.PAWN, piece == chess.QUEEN, piece == chess.KING]


def centipawn_losses(engine: chess.engine.SimpleEngine, board: chess.Board, moves: list[chess.Move], depth: int) -> list[int]:
    """Perte en centipions de chaque coup par rapport au meilleur coup de Stockfish, du point de vue du joueur au trait."""
    limit = chess.engine.Limit(depth=depth)

    def score(b: chess.Board) -> int:
        if b.is_checkmate():
            return MATE if b.turn != board.turn else -MATE
        if b.is_game_over():
            return 0
        return engine.analyse(b, limit)["score"].pov(board.turn).score(mate_score=MATE)

    best, cache = score(board), {}
    for m in moves:
        if m not in cache:
            after = board.copy()
            after.push(m)
            cache[m] = min(CPL_CAP, max(0, best - score(after)))
    return [cache[m] for m in moves]


def sample(path: str, per_player: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Positions tirées au hasard par joueur (hors ouverture et hors identifiant générique), avec le nom du joueur."""
    rng = np.random.default_rng(seed)
    out = {}
    with h5py.File(path) as f:
        names = json.loads(f.attrs["players"]) if "players" in f.attrs else list(PLAYERS)
        player, ply = f["player_id"][:], f["ply"][:]
        for p, name in enumerate(names):
            rows = np.flatnonzero((player == p) & (ply >= OPENING_PLIES)) if p != GENERIC else []
            if len(rows):
                rows = np.sort(rng.choice(rows, min(per_player, len(rows)), replace=False))
                out[name] = {k: f[k][rows] for k in ("boards", "legal_mask", "target")}
    return out


@torch.no_grad()
def model_moves(model: torch.nn.Module, data: dict[str, np.ndarray], model_id: int, batch_size: int = 256) -> np.ndarray:
    """Indice du coup légal préféré du modèle, conditionné sur `model_id`."""
    boards, masks = torch.from_numpy(data["boards"]), torch.from_numpy(data["legal_mask"])
    picks = []
    for i in range(0, len(boards), batch_size):
        ids = torch.full((len(boards[i : i + batch_size]),), model_id, dtype=torch.long)
        logits = model(boards[i : i + batch_size], ids)
        logits = (logits[0] if isinstance(logits, tuple) else logits).float()
        picks.append(logits.masked_fill(~masks[i : i + batch_size], float("-inf")).argmax(1))
    return torch.cat(picks).numpy()


def bench(
    model: torch.nn.Module, players: list[str], path: str, per_player: int = 300, engine=None, depth: int = 10, seed: int = 0
) -> dict:
    game, generic = ChessGame(), players.index(PLAYERS[GENERIC])
    result = {"players": {}}
    for name, data in sample(path, per_player, seed).items():
        picks = {"reel": data["target"], "modele": model_moves(model, data, players.index(name)),
                 "generique": model_moves(model, data, generic)}
        rows = {s: [] for s in SOURCES}
        losses = {s: [] for s in SOURCES}
        for k, planes in enumerate(data["boards"]):
            board = board_from_planes(planes)
            moves = {s: game.index_to_move(int(picks[s][k]), board) for s in SOURCES}
            for s in SOURCES:
                rows[s].append(features(board, moves[s]))
            if engine:
                for s, loss in zip(SOURCES, centipawn_losses(engine, board, list(moves.values()), depth)):
                    losses[s].append(loss)
        entry = {"positions": len(data["target"])}
        for s in SOURCES:
            entry[s] = dict(zip(FEATURES, np.mean(rows[s], axis=0).tolist()))
            if engine:
                entry[s]["cpl"] = float(np.mean(losses[s]))
        for s in ("modele", "generique"):
            entry[f"distance_{s}"] = float(sum(abs(entry[s][f] - entry["reel"][f]) for f in FEATURES))
        result["players"][name] = entry

    names = list(result["players"])
    p = result["players"]
    result["contraste"] = float(np.mean([sum(abs(p[a]["reel"][f] - p[b]["reel"][f]) for f in FEATURES) for a, b in combinations(names, 2)])) if len(names) > 1 else None
    result["modele_plus_proche"] = sum(e["distance_modele"] < e["distance_generique"] for e in p.values())
    return result


def report(result: dict) -> str:
    p = result["players"]
    has_cpl = all("cpl" in e["reel"] for e in p.values())
    lines = [f"{'joueur':<12}{'source':<11}" + "".join(f"{f:>9}" for f in FEATURES) + (f"{'cpl':>8}" if has_cpl else "") + f"{'dist. au réel':>15}"]
    for name, e in p.items():
        for s in SOURCES:
            values = "".join(f"{100 * e[s][f]:>8.1f}%" for f in FEATURES)
            cpl = f"{e[s]['cpl']:>8.0f}" if has_cpl else ""
            dist = "" if s == "reel" else f"{100 * e['distance_' + s]:>14.1f}"
            lines.append(f"{name if s == 'reel' else '':<12}{s:<11}{values}{cpl}{dist:>15}")
    lines += ["", f"Modèle conditionné plus proche du réel que le générique : {result['modele_plus_proche']}/{len(p)} joueurs",
              f"Contraste moyen entre les profils réels (L1, en points) : {100 * result['contraste']:.1f}" if result["contraste"] is not None else ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="dossier de run de train.py")
    p.add_argument("h5", help="HDF5 de build_dataset.py, en général le split test")
    p.add_argument("--per-player", type=int, default=300)
    p.add_argument("--depth", type=int, default=10, help="profondeur de Stockfish")
    p.add_argument("--stockfish", help="chemin du binaire (sinon celui du PATH ; sans Stockfish, pas de perte en centipions)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="fichier JSON des résultats")
    a = p.parse_args(argv)

    model, players = load_model(a.model)
    path = a.stockfish or shutil.which("stockfish")
    engine = chess.engine.SimpleEngine.popen_uci(path) if path else None
    if engine:
        engine.configure({"Threads": 1})  # un seul fil : évaluations reproductibles
    else:
        print("Stockfish introuvable : perte en centipions non calculée")
    try:
        result = bench(model, players, a.h5, a.per_player, engine, a.depth, a.seed)
    finally:
        if engine:
            engine.quit()
    result["model"], result["data"] = str(a.model), str(a.h5)
    print(report(result))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
