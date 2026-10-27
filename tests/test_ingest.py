import random

import chess
import chess.pgn

from chessstyle.ingest import ingest, stats
from chessstyle.players import PLAYER_ID


def make_pgn(white="A", black="B", plies=60, seed=0, **headers) -> str:
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        board.push(rng.choice(list(board.legal_moves)))
    game = chess.pgn.Game.from_board(board)
    game.headers.update({"White": white, "Black": black, "Date": "2000.01.01", **headers})
    return str(game) + "\n\n"


LICHESS = dict(WhiteElo="2300", BlackElo="2250", TimeControl="600+0")


def run(tmp_path, source, *pgns):
    path = tmp_path / "x.pgn"
    path.write_text("".join(pgns), encoding="utf-8")
    return ingest([path], source)


def test_lichess_filters(tmp_path):
    good = make_pgn(seed=1, **LICHESS)
    bad = [
        make_pgn(seed=2, **{**LICHESS, "BlackElo": "2100"}),  # Elo trop bas
        make_pgn(seed=3, **{**LICHESS, "TimeControl": "60+0"}),  # bullet
        make_pgn(seed=4, plies=30, **LICHESS),  # trop courte
        make_pgn(seed=5, WhiteElo="2300", BlackElo="2250", EventType="blitz"),  # format Elite DB
    ]
    df = run(tmp_path, "lichess", good, *bad)
    assert len(df) == 2 and set(df.player_id) == {0} and set(df.color) == {True, False}


def test_lichess_elite_format_without_timecontrol(tmp_path):
    df = run(tmp_path, "lichess", make_pgn(seed=1, WhiteElo="2300", BlackElo="2250", EventType="rapid"))
    assert len(df) == 2


def test_dedup_and_stable_game_id(tmp_path):
    pgn = make_pgn(seed=1, **LICHESS)
    once = run(tmp_path, "lichess", pgn)
    twice = run(tmp_path, "lichess", pgn, pgn)
    assert len(twice) == len(once) == 2
    assert list(once.game_id) == list(twice.game_id)


def test_icons_label_each_side_with_its_own_player_id(tmp_path):
    df = run(
        tmp_path,
        "icons",
        make_pgn("Tal, Mikhail", "Petrosian, Tigran V", seed=1),
        make_pgn("Tal, Mikhail", "Nobody, X", seed=2),
        make_pgn("Nobody, X", "Nobody, Y", seed=3),
        make_pgn("Tal, Mikhail", "Nobody, X", seed=4, Event="Blitz 5 min"),
    )
    pairs = sorted(zip(df.color, df.player_id))
    assert pairs == sorted([(True, PLAYER_ID["Tal"]), (False, PLAYER_ID["Petrosian"]), (True, PLAYER_ID["Tal"])])


def test_stats_counts_only_the_imitated_sides_moves(tmp_path):
    df = run(tmp_path, "lichess", make_pgn(plies=61, seed=1, **LICHESS))
    assert stats(df).positions.iloc[0] == 31 + 30  # Blancs 31 coups, Noirs 30
