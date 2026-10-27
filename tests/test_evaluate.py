import json

import chess
import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chessstyle.build_dataset import FIELDS, H5Writer, encode_side
from chessstyle.encoding import ChessGame
from chessstyle.evaluate import PHASES, load_model, main, phase_of, score, summarize
from chessstyle.model import ChessNet
from chessstyle.players import PLAYERS
from chessstyle.train import save

GAME = ChessGame()


def encoded(fen: str) -> np.ndarray:
    return GAME.encode(chess.Board(fen))[None]


@pytest.mark.parametrize(
    "fen, ply, phase",
    [
        (chess.STARTING_FEN, 0, "ouverture"),
        (chess.STARTING_FEN, 19, "ouverture"),
        (chess.STARTING_FEN, 20, "milieu"),
        ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 10, "ouverture"),  # coups 1 à 10 : ouverture même sans pièces
        ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 60, "finale"),
        ("r1b1k2r/8/8/8/8/8/8/R1B1K2R w - - 0 1", 60, "finale"),  # 6 pièces
        ("rnb1k2r/8/8/8/8/8/8/R1B1K2R w - - 0 1", 60, "milieu"),  # 7 pièces
    ],
)
def test_phase_of(fen, ply, phase):
    assert PHASES[phase_of(encoded(fen), np.array([ply]))[0]] == phase


class StyleOracle(torch.nn.Module):
    """Joue toujours le coup d'indice player_id : chaque joueur n'est bien prédit que par lui-même."""

    def forward(self, board, player_id):
        logits = torch.zeros(len(board), 4672)
        logits[torch.arange(len(board)), player_id] = 10.0
        return logits


@pytest.fixture
def synthetic(tmp_path):
    """Positions des joueurs 1 à 3, dont le coup joué est l'indice du joueur, plus des coups d'ouverture qui le contredisent."""
    path = tmp_path / "test.h5"
    per_player, opening = 50, 30
    players = np.repeat([1, 2, 3], per_player + opening)
    ply = np.tile(np.r_[np.full(per_player, 40), np.arange(opening) % 20], 3)
    target = np.where(ply >= 20, players, 5)
    n = len(players)
    mask = np.zeros((n, 4672), bool)
    mask[:, :10] = True
    with h5py.File(path, "w") as f:
        for k, (shape, dtype) in FIELDS.items():
            f.create_dataset(k, shape=(n, *shape), dtype=dtype)
        f["legal_mask"][:], f["target"][:], f["player_id"][:], f["ply"][:] = mask, target, players, ply
        f.attrs["players"] = json.dumps(PLAYERS)
        f.attrs["encoding_version"] = 1
    return str(path), per_player, opening


def test_cross_matrix_diagonal(synthetic):
    path, per_player, _ = synthetic
    s = score(StyleOracle(), path, PLAYERS, batch_size=7)  # batch qui ne divise pas les données
    result = summarize(s, PLAYERS)
    cross = result["cross"]
    assert cross["cols"] == ["Tal", "Petrosian", "Capablanca"]
    assert cross["rows"] == PLAYERS
    for i, row in enumerate(cross["top1"]):
        assert row == [1.0 if i == j else 0.0 for j in (1, 2, 3)]
    assert cross["diagonal_wins_top1"] == cross["diagonal_wins_nll"] == 3
    assert result["overall"] == {"positions": 3 * per_player, "top1": 1.0, "top3": 1.0, "nll": pytest.approx(0, abs=1e-3)}


def test_opening_is_measured_apart(synthetic):
    path, per_player, opening = synthetic
    tal = summarize(score(StyleOracle(), path, PLAYERS), PLAYERS)["players"]["Tal"]
    assert tal["positions"] == per_player and tal["top1"] == 1.0
    # Le coup joué en ouverture est 2e, à égalité avec les autres coups légaux : hors du top-1, dans le top-3
    assert tal["phases"]["ouverture"] == {"positions": opening, "top1": 0.0, "top3": 1.0, "nll": pytest.approx(10, abs=0.01)}
    assert tal["gain_vs_generique"] == 1.0


def test_data_players_are_matched_by_name(synthetic):
    path, _, _ = synthetic
    reordered = ["generique", "Capablanca", "Tal", "Petrosian"]  # le modèle a une autre liste que les données
    s = score(StyleOracle(), path, reordered)
    assert set(np.unique(s["player"])) == {1, 2, 3}
    assert summarize(s, reordered)["cross"]["cols"] == ["Capablanca", "Tal", "Petrosian"]


@pytest.fixture
def run(tmp_path):
    model = ChessNet(channels=8, blocks=1)
    save(model, {"model": {"num_players": len(PLAYERS), "channels": 8, "blocks": 1}, "encoding_version": 1}, tmp_path / "run")
    return tmp_path / "run", model


def test_load_model_reads_a_train_run(run):
    path, model = run
    loaded, players = load_model(str(path))
    x = torch.from_numpy(encoded(chess.STARTING_FEN))
    ids = torch.tensor([2])
    model.eval()
    assert players == PLAYERS
    assert torch.equal(loaded(x, ids), model(x, ids))


def test_load_model_refuses_another_encoding(run):
    path, _ = run
    (path / "config.yaml").write_text((path / "config.yaml").read_text().replace("encoding_version: 1", "encoding_version: 2"))
    with pytest.raises(ValueError, match="v2"):
        load_model(str(path))


def test_command_line_on_real_positions(run, tmp_path, capsys):
    path, _ = run
    moves = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c8b7 d2d4 f8e8 b1d2 e7f8 a2a4 h7h6"
    writer = H5Writer(tmp_path / "test.h5", {"players": json.dumps(PLAYERS), "encoding_version": 1})
    writer.add(encode_side(moves, chess.WHITE, 1, "g1"))
    writer.add(encode_side(moves, chess.BLACK, 2, "g1"))
    writer.close()

    result = main([str(path), str(tmp_path / "test.h5"), "--out", str(tmp_path / "eval.json")])
    assert json.loads((tmp_path / "eval.json").read_text()) == json.loads(json.dumps(result))
    assert result["overall"]["positions"] == 4  # demi-coups 20 à 23
    assert result["players"]["Tal"]["phases"]["ouverture"]["positions"] == 10
    assert "Matrice croisée" in capsys.readouterr().out
