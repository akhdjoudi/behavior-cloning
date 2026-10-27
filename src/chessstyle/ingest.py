"""PGN -> parties filtrées et dédupliquées, avec game_id et player_id.

Responsable : rôle 1. Voir la section correspondante du plan de projet.

Sortie : un Parquet avec une ligne par côté imité d'une partie
(game_id, player_id, color, moves en UCI séparés par des espaces).
Les positions utiles sont celles où `color` a le trait : coups d'indice pair pour les Blancs, impair pour les Noirs.

    python -m chessstyle.ingest lichess data/raw/elite.pgn.zst -o data/interim/lichess.parquet
    python -m chessstyle.ingest icons data/raw/Tal.pgn data/raw/Petrosian.pgn -o data/interim/icons.parquet
"""

import argparse
import hashlib
import io
import warnings
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

from chessstyle.players import GENERIC, PLAYER_ID, PLAYERS

MIN_ELO = 2200
MIN_PLIES = 40  # 20 coups
MIN_SECONDS = 480  # base + 40 * incrément : seuil « rapide » de Lichess
SLOW_EVENTS = ("blitz", "bullet")
MIN_GAMES = 1000  # en dessous, le style du joueur sera mal appris


def _open(path: Path):
    if path.suffix == ".zst":
        import zstandard

        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(path.open("rb")), encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _elo(headers, key: str) -> int:
    try:
        return int(headers.get(key, ""))
    except ValueError:
        return 0


def _estimated_seconds(time_control: str) -> int | None:
    try:
        base, _, inc = time_control.partition("+")
        return int(base) + 40 * int(inc or 0)
    except ValueError:
        return None  # "-", "?" ou absent


def _surname(name: str) -> str:
    return name.split(",")[0].strip().lower()


def keep_lichess(game: chess.pgn.Game, plies: int) -> bool:
    h = game.headers
    if h.get("Variant", "Standard") != "Standard" or plies < MIN_PLIES:
        return False
    if min(_elo(h, "WhiteElo"), _elo(h, "BlackElo")) < MIN_ELO:
        return False
    seconds = _estimated_seconds(h.get("TimeControl", ""))
    return seconds is not None and seconds >= MIN_SECONDS


def keep_icon(game: chess.pgn.Game) -> bool:
    event = game.headers.get("Event", "").lower()
    return game.headers.get("Variant", "Standard") == "Standard" and not any(w in event for w in SLOW_EVENTS)


def game_id(moves: str, h) -> str:
    key = "|".join([moves, h.get("Date", "?"), h.get("White", "?"), h.get("Black", "?")])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def sides(game: chess.pgn.Game, source: str) -> list[tuple[bool, int]]:
    """(couleur, player_id) de chaque côté imité de la partie."""
    if source == "lichess":
        return [(chess.WHITE, GENERIC), (chess.BLACK, GENERIC)]
    out = []
    for color, key in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        pid = next((i for n, i in PLAYER_ID.items() if i != GENERIC and n.lower() == _surname(game.headers.get(key, ""))), None)
        if pid is not None:
            out.append((color, pid))
    return out


def ingest(paths: list[Path], source: str) -> pd.DataFrame:
    seen: set[str] = set()
    rows = []
    for path in paths:
        with _open(path) as f:
            while (game := chess.pgn.read_game(f)) is not None:
                if game.errors:
                    continue
                moves = " ".join(m.uci() for m in game.mainline_moves())
                plies = moves.count(" ") + 1 if moves else 0
                if not (keep_lichess(game, plies) if source == "lichess" else keep_icon(game)):
                    continue
                gid = game_id(moves, game.headers)
                if gid in seen:
                    continue
                seen.add(gid)
                for color, pid in sides(game, source):
                    rows.append({"game_id": gid, "player_id": pid, "color": color, "moves": moves, "source": source})
    return pd.DataFrame(rows, columns=["game_id", "player_id", "color", "moves", "source"])


def stats(df: pd.DataFrame) -> pd.DataFrame:
    """Parties et positions (coups joués par le joueur imité) par player_id."""
    plies = df["moves"].str.count(" ") + 1
    df = df.assign(positions=(plies + (df["color"] == chess.WHITE)) // 2)
    return df.groupby("player_id").agg(games=("game_id", "count"), positions=("positions", "sum"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["lichess", "icons"])
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    args = p.parse_args()
    df = ingest(args.paths, args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    if df.empty:
        print("aucune partie retenue")
    else:
        table = stats(df)
        print(table)
        if args.source == "icons":
            for pid, games in table["games"].items():
                if games < MIN_GAMES:
                    warnings.warn(f"{PLAYERS[pid]} : seulement {games} parties (< {MIN_GAMES}), style mal appris")
