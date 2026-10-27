"""Parties -> fichiers HDF5 (boards, legal_mask, target, meta), split par game_id.

Responsable : rôle 2. Voir la section correspondante du plan de projet.

Entrée : le Parquet de `ingest.py` (game_id, player_id, color, moves, source).
Sortie : `train.h5`, `val.h5`, `test.h5` dans le dossier donné, une ligne par coup joué par le côté imité :

    boards      (N, 18, 8, 8) uint8   contrat d'encodage v1
    legal_mask  (N, 4672)     bool
    target      (N,)          int16   index du coup joué
    player_id   (N,)          uint8
    game_id     (N,)          S16     pour regrouper par partie à l'évaluation
    ply         (N,)          uint16  demi-coup de la position (0 = position initiale)

Le split se fait par partie (hash du game_id), jamais par position : les deux côtés d'une partie
tombent dans le même fichier. Les parties sont mélangées avant l'écriture pour que chaque chunk
de 4 096 lignes mélange les joueurs.

    python -m chessstyle.build_dataset data/interim/lichess.parquet -o data/processed/lichess
    python -m chessstyle.build_dataset data/interim/icons.parquet -o data/processed/icons
    python -m chessstyle.build_dataset data/interim/icons.parquet -o data/sample --max-games 200
"""

import argparse
import hashlib
import json
import os
import multiprocessing
from pathlib import Path

import chess
import h5py
import numpy as np
import pandas as pd

from chessstyle.encoding import ENCODING_VERSION, NUM_ACTIONS, NUM_PLANES, ChessGame
from chessstyle.players import PLAYERS

SPLITS = (("train", 0.90), ("val", 0.05), ("test", 0.05))
CHUNK = 4096

FIELDS = {
    "boards": ((NUM_PLANES, 8, 8), np.uint8),
    "legal_mask": ((NUM_ACTIONS,), np.bool_),
    "target": ((), np.int16),
    "player_id": ((), np.uint8),
    "game_id": ((), "S16"),
    "ply": ((), np.uint16),
}

_GAME = ChessGame()


def split_of(game_id: str) -> str:
    """Split stable d'une partie, indépendant de l'ordre et du contenu du fichier."""
    u = int(hashlib.sha1(f"split|{game_id}".encode()).hexdigest()[:8], 16) / 2**32
    cumulative = 0.0
    for name, share in SPLITS:
        cumulative += share
        if u < cumulative:
            return name
    return SPLITS[-1][0]


def encode_side(moves: str, color: bool, player_id: int, game_id: str) -> dict[str, np.ndarray]:
    """Toutes les positions où `color` a le trait, avec le coup qu'il a joué."""
    rows = {k: [] for k in FIELDS}
    board = chess.Board()
    for ply, uci in enumerate(moves.split()):
        move = chess.Move.from_uci(uci)
        if board.turn == color:
            rows["boards"].append(_GAME.encode(board))
            rows["legal_mask"].append(_GAME.legal_mask(board))
            rows["target"].append(_GAME.move_to_index(move, board))
            rows["ply"].append(ply)
        board.push(move)
    n = len(rows["target"])
    rows["player_id"] = [player_id] * n
    rows["game_id"] = [game_id] * n
    return {k: np.asarray(rows[k], dtype=dtype).reshape((n, *shape)) for k, (shape, dtype) in FIELDS.items()}


def _encode_row(row: tuple) -> dict[str, np.ndarray]:
    return encode_side(*row)


class H5Writer:
    """Écrit par blocs de CHUNK lignes dans des datasets redimensionnables et compressés."""

    def __init__(self, path: Path, attrs: dict):
        self.file = h5py.File(path, "w")
        for k, (shape, dtype) in FIELDS.items():
            self.file.create_dataset(
                k, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype,
                chunks=(CHUNK, *shape), compression="gzip", compression_opts=1,
            )
        self.file.attrs.update(attrs)
        self.buffer: list[dict[str, np.ndarray]] = []
        self.buffered = 0
        self.games = 0

    def add(self, rows: dict[str, np.ndarray]) -> None:
        self.buffer.append(rows)
        self.buffered += len(rows["target"])
        self.games += 1
        if self.buffered >= CHUNK:
            self.flush()

    def flush(self) -> None:
        if not self.buffered:
            return
        for k in FIELDS:
            block = np.concatenate([r[k] for r in self.buffer])
            ds = self.file[k]
            start = ds.shape[0]
            ds.resize(start + len(block), axis=0)
            ds[start:] = block
        self.buffer, self.buffered = [], 0

    def close(self) -> int:
        self.flush()
        n = self.file["target"].shape[0]
        self.file.attrs["num_positions"] = n
        self.file.attrs["num_sides"] = self.games
        self.file.close()
        return n


def build(parquets: list[Path], out_dir: Path, max_games: int | None = None, workers: int = 1, seed: int = 0) -> dict[str, int]:
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    games = df["game_id"].drop_duplicates().sample(frac=1, random_state=seed)
    if max_games is not None:
        games = games.iloc[:max_games]
    order = pd.Series(range(len(games)), index=games.values)
    df = df[df["game_id"].isin(order.index)]
    df = df.assign(_order=df["game_id"].map(order)).sort_values(["_order", "color"], ascending=[True, False])

    out_dir.mkdir(parents=True, exist_ok=True)
    attrs = {
        "encoding_version": ENCODING_VERSION,
        "players": json.dumps(PLAYERS, ensure_ascii=False),
        "sources": json.dumps([str(p) for p in parquets]),
    }
    writers = {name: H5Writer(out_dir / f"{name}.h5", {**attrs, "split": name}) for name, _ in SPLITS}
    splits = df["game_id"].map(split_of).tolist()
    rows = list(zip(df["moves"], df["color"].astype(bool), df["player_id"].astype(int), df["game_id"]))

    if workers > 1:
        # spawn : même comportement sous Linux et Windows, pas de fork d'un processus multi-thread
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for split, encoded in zip(splits, pool.imap(_encode_row, rows, chunksize=64)):
                writers[split].add(encoded)
    else:
        for split, row in zip(splits, rows):
            writers[split].add(_encode_row(row))
    return {name: w.close() for name, w in writers.items()}


def summary(path: Path) -> pd.Series:
    """Positions par joueur dans un fichier HDF5."""
    with h5py.File(path) as f:
        ids = f["player_id"][:]
    counts = pd.Series(ids).value_counts().sort_index()
    return counts.rename(index=lambda i: PLAYERS[i])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("parquets", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True, help="dossier de sortie (train.h5, val.h5, test.h5)")
    p.add_argument("--max-games", type=int, help="ne garde que N parties tirées au hasard (échantillon)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sizes = build(args.parquets, args.out, args.max_games, args.workers, args.seed)
    for name, n in sizes.items():
        print(f"{name}: {n} positions")
        if n:
            print(summary(args.out / f"{name}.h5").to_string(), "\n")
