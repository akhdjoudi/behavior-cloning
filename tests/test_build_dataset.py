import random

import chess
import h5py
import numpy as np
import pandas as pd
import pytest

from chessstyle.build_dataset import FIELDS, build, encode_side, split_of
from chessstyle.encoding import ENCODING_VERSION, ChessGame
from chessstyle.players import GENERIC, PLAYER_ID


def random_moves(plies: int, seed: int) -> str:
    rng, board = random.Random(seed), chess.Board()
    moves = []
    while len(moves) < plies and not board.is_game_over():
        move = rng.choice(list(board.legal_moves))
        moves.append(move.uci())
        board.push(move)
    return " ".join(moves)


def make_parquet(tmp_path, games: int = 60) -> tuple:
    """Comme la sortie d'ingest.py : une ligne par côté imité, parfois les deux côtés d'une même partie."""
    rows = []
    for i in range(games):
        moves, gid = random_moves(40 + i % 7, seed=i), f"{i:016x}"
        rows.append({"game_id": gid, "player_id": PLAYER_ID["Tal"], "color": True, "moves": moves, "source": "icons"})
        if i % 2:
            rows.append({"game_id": gid, "player_id": PLAYER_ID["Petrosian"], "color": False, "moves": moves, "source": "icons"})
    df = pd.DataFrame(rows)
    path = tmp_path / "icons.parquet"
    df.to_parquet(path, index=False)
    return path, df


def read(path) -> dict[str, np.ndarray]:
    with h5py.File(path) as f:
        return {k: f[k][:] for k in FIELDS}


def test_encode_side_replays_the_played_moves():
    moves = random_moves(41, seed=3)
    game = ChessGame()
    for color in (chess.WHITE, chess.BLACK):
        rows = encode_side(moves, color, GENERIC, "0" * 16)
        assert len(rows["target"]) == (41 + color) // 2
        assert set(rows["ply"] % 2) == {0 if color else 1}
        played = moves.split()
        for i, ply in enumerate(rows["ply"]):
            board = chess.Board()
            for uci in played[:ply]:
                board.push_uci(uci)
            assert np.array_equal(rows["boards"][i], game.encode(board))
            assert rows["legal_mask"][i, rows["target"][i]]
            assert game.index_to_move(int(rows["target"][i]), board).uci() == played[ply]


def test_shapes_dtypes_and_metadata(tmp_path):
    path, _ = make_parquet(tmp_path)
    build([path], tmp_path / "out")
    with h5py.File(tmp_path / "out" / "train.h5") as f:
        n = f["target"].shape[0]
        assert n > 0 and f.attrs["num_positions"] == n
        assert f.attrs["encoding_version"] == ENCODING_VERSION and f.attrs["split"] == "train"
        for k, (shape, dtype) in FIELDS.items():
            assert f[k].shape == (n, *shape) and f[k].dtype == np.dtype(dtype)


def test_every_position_is_kept_once_and_split_by_game(tmp_path):
    path, df = make_parquet(tmp_path)
    sizes = build([path], tmp_path / "out")
    expected = sum(len(encode_side(m, c, p, g)["target"]) for m, c, p, g in zip(df.moves, df.color, df.player_id, df.game_id))
    assert sum(sizes.values()) == expected

    games = {name: set(read(tmp_path / "out" / f"{name}.h5")["game_id"]) for name in sizes}
    assert not games["train"] & games["val"] and not games["train"] & games["test"] and not games["val"] & games["test"]
    for name, ids in games.items():
        assert all(split_of(g.decode()) == name for g in ids)


def test_player_ids_are_kept(tmp_path):
    path, df = make_parquet(tmp_path)
    build([path], tmp_path / "out")
    data = read(tmp_path / "out" / "train.h5")
    black = data["ply"] % 2 == 1
    assert set(data["player_id"][black]) <= {PLAYER_ID["Petrosian"]}
    assert set(data["player_id"][~black]) <= {PLAYER_ID["Tal"]}


def test_parallel_build_matches_sequential(tmp_path):
    path, _ = make_parquet(tmp_path, games=20)
    build([path], tmp_path / "seq", workers=1)
    build([path], tmp_path / "par", workers=2)
    for name in ("train", "val", "test"):
        a, b = read(tmp_path / "seq" / f"{name}.h5"), read(tmp_path / "par" / f"{name}.h5")
        assert all(np.array_equal(a[k], b[k]) for k in FIELDS)


def test_max_games(tmp_path):
    path, _ = make_parquet(tmp_path)
    build([path], tmp_path / "out", max_games=5)
    ids = set().union(*(read(tmp_path / "out" / f"{s}.h5")["game_id"] for s in ("train", "val", "test")))
    assert len(ids) == 5


def test_split_proportions():
    counts = pd.Series([split_of(f"{i:016x}") for i in range(20_000)]).value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.90, abs=0.01)
    assert counts["val"] == pytest.approx(0.05, abs=0.01)
    assert counts["test"] == pytest.approx(0.05, abs=0.01)


def test_train_py_can_read_the_output(tmp_path):
    pytest.importorskip("torch")
    from chessstyle.train import load_h5

    path, _ = make_parquet(tmp_path)
    build([path], tmp_path / "out")
    data = load_h5(str(tmp_path / "out" / "train.h5"))
    assert data["board"].shape[1:] == (18, 8, 8) and data["mask"].shape[1] == 4672
    assert data["mask"][range(len(data["target"])), data["target"]].all()
