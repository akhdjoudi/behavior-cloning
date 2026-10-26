import random

import chess
import numpy as np
import pytest

from chessstyle.encoding import NUM_ACTIONS, ChessGame

game = ChessGame()


def random_positions(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    positions = []
    while len(positions) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            positions.append(board)
    return positions


def assert_round_trip(board: chess.Board) -> None:
    indices = set()
    for move in board.legal_moves:
        idx = game.move_to_index(move, board)
        assert 0 <= idx < NUM_ACTIONS
        assert game.index_to_move(idx, board) == move, f"{move.uci()} dans {board.fen()}"
        indices.add(idx)
    assert len(indices) == board.legal_moves.count(), "deux coups légaux partagent le même index"


def test_round_trip_on_random_positions():
    for board in random_positions(1000):
        assert_round_trip(board)


@pytest.mark.parametrize(
    "fen, uci",
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3"),
    ],
)
def test_castling_and_en_passant(fen, uci):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    assert game.index_to_move(game.move_to_index(move, board), board) == move
    assert_round_trip(board)


@pytest.mark.parametrize(
    "fen",
    [
        "1n2k3/2P5/8/8/8/8/8/4K3 w - - 0 1",  # Blancs : promotion tout droit et avec prise
        "4k3/8/8/8/8/8/2p5/1N2K3 b - - 0 1",  # Noirs : idem
    ],
)
def test_all_promotions(fen):
    board = chess.Board(fen)
    promotions = [m for m in board.legal_moves if m.promotion]
    assert len(promotions) == 8  # 2 cases d'arrivée x 4 pièces
    assert_round_trip(board)


def test_initial_position_is_the_same_for_both_sides():
    white_view = game.encode(chess.Board())
    black_view = game.encode(chess.Board().mirror())  # mêmes pièces, couleurs inversées, trait aux Noirs
    assert np.array_equal(white_view, black_view)


def test_encode_shape_and_planes():
    planes = game.encode(chess.Board())
    assert planes.shape == (18, 8, 8)
    assert planes.dtype == np.uint8
    assert planes[0].sum() == 8 and planes[6].sum() == 8  # pions
    assert planes[12:16].min() == 1  # quatre droits de roque
    assert planes[16].sum() == 0  # pas de prise en passant


def test_reference_index_e2e4_and_e7e5():
    board = chess.Board()
    assert game.move_to_index(chess.Move.from_uci("e2e4"), board) == 76
    board.push_uci("e2e4")
    assert game.move_to_index(chess.Move.from_uci("e7e5"), board) == 76


def test_legal_mask_matches_legal_moves():
    for board in random_positions(200, seed=1):
        assert game.legal_mask(board).sum() == board.legal_moves.count()
