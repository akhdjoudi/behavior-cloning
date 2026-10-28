"""style_bench : le plateau reconstruit depuis les plans donne les mêmes coups légaux, les mesures sont justes."""

import random
import shutil

import chess
import chess.engine
import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chessstyle.build_dataset import CHUNK, FIELDS
from chessstyle.encoding import ChessGame
from chessstyle.model import ChessNet
from chessstyle.style_bench import FEATURES, bench, board_from_planes, centipawn_losses, features
from test_overfit import random_positions

GAME = ChessGame()
STOCKFISH = shutil.which("stockfish")


def test_board_from_planes_keeps_legal_moves_for_both_sides():
    rng, board, checked = random.Random(1), chess.Board(), 0
    for _ in range(400):
        if board.is_game_over():
            board = chess.Board()
        rebuilt = board_from_planes(GAME.encode(board))
        assert (GAME.encode(rebuilt) == GAME.encode(board)).all()
        assert (GAME.legal_mask(rebuilt) == GAME.legal_mask(board)).all()  # mêmes coups, même position canonique
        checked += board.turn == chess.BLACK
        board.push(rng.choice(list(board.legal_moves)))
    assert checked > 50  # le miroir des Noirs est bien couvert


def test_board_from_planes_en_passant_and_castling():
    board = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1")
    rebuilt = board_from_planes(GAME.encode(board))
    assert rebuilt.ep_square == chess.D6 and rebuilt.has_kingside_castling_rights(chess.BLACK)
    assert rebuilt.is_en_passant(chess.Move.from_uci("e5d6"))


def test_features():
    start = chess.Board()
    assert dict(zip(FEATURES, features(start, chess.Move.from_uci("e2e4")))) == {
        "capture": False, "echec": False, "pion": True, "dame": False, "roi": False}
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    assert dict(zip(FEATURES, features(board, chess.Move.from_uci("d8h4")))) == {
        "capture": False, "echec": True, "pion": False, "dame": True, "roi": False}
    assert features(chess.Board("4k3/8/8/8/8/8/4p3/4K3 w - - 0 1"), chess.Move.from_uci("e1e2"))[0]  # le roi prend


@pytest.mark.skipif(STOCKFISH is None, reason="Stockfish absent")
def test_centipawn_loss_best_move_is_free_and_blunder_is_costly():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    try:
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")  # Td8 est mat
        mate, blunder = chess.Move.from_uci("d1d8"), chess.Move.from_uci("g1h1")
        assert centipawn_losses(engine, board, [mate], depth=6) == [0]
        assert centipawn_losses(engine, board, [blunder], depth=6)[0] > 500
    finally:
        engine.quit()


def test_bench_end_to_end(tmp_path):
    data = random_positions(600)
    path = tmp_path / "test.h5"
    keys = {"boards": "board", "legal_mask": "mask", "target": "target", "player_id": "player_id"}
    with h5py.File(path, "w") as f:
        for k, (shape, dtype) in FIELDS.items():
            f.create_dataset(k, shape=(600, *shape), dtype=dtype, chunks=(CHUNK, *shape), maxshape=(None, *shape))
            if k in keys:
                f[k][:] = data[keys[k]].numpy()
        f["ply"][:] = 40  # hors ouverture

    model = ChessNet(channels=16, blocks=1).eval()
    players = ["generique", "Tal", "Petrosian", "Capablanca", "Fischer", "Kasparov", "Carlsen"]
    result = bench(model, players, str(path), per_player=20)
    assert 0 < len(result["players"]) <= 6 and "generique" not in result["players"]
    for e in result["players"].values():
        assert e["positions"] <= 20
        assert all(0 <= e[s][f] <= 1 for s in ("reel", "modele", "generique") for f in FEATURES)
    assert 0 <= result["modele_plus_proche"] <= len(result["players"])
