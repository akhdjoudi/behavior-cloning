"""Test de sanité : le modèle doit surapprendre 1 000 positions (top-1 proche de 100 %), sinon il y a un bug."""

import random

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chessstyle.encoding import ChessGame
from chessstyle.train import evaluate, train


def random_positions(n: int, seed: int = 0) -> dict:
    rng, game = random.Random(seed), ChessGame()
    boards, masks, targets = [], [], []
    board = chess.Board()
    while len(targets) < n:
        if board.is_game_over():
            board = chess.Board()
        move = rng.choice(list(board.legal_moves))
        boards.append(game.encode(board))
        masks.append(game.legal_mask(board))
        targets.append(game.move_to_index(move, board))
        board.push(move)
    return {
        "board": torch.from_numpy(np.stack(boards)),
        "mask": torch.from_numpy(np.stack(masks)),
        "target": torch.tensor(targets),
        "player_id": torch.randint(0, 7, (n,)),
    }


def test_overfit_1000_positions(tmp_path):
    data = random_positions(1000)
    model = train(data, str(tmp_path), epochs=40, batch_size=100, lr=3e-3, model_kwargs={"channels": 32, "blocks": 3})
    _, top1 = evaluate(model, data, "cpu")
    assert top1 > 0.95
    assert {"model.pt", "config.yaml", "players.json"} <= {p.name for p in tmp_path.iterdir()}
