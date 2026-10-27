import chess
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from chessstyle.inference import Predictor
from chessstyle.model import ChessNet, export_dummy
from chessstyle.players import PLAYER_ID, PLAYERS
from chessstyle.server import create_app
from chessstyle.train import save

PROMOTION = "7k/P7/8/8/8/8/8/K7 w - - 0 1"


@pytest.fixture(scope="module")
def client():
    torch.manual_seed(0)
    return TestClient(create_app(Predictor(), web_dist=None))


def test_info(client):
    info = client.get("/api/info").json()
    assert info["players"] == PLAYERS and info["trained"] is False and info["encoding_version"] == 1


def test_predict_returns_every_legal_move_sorted(client):
    fen = chess.Board().fen()
    res = client.post("/api/predict", json={"fen": fen, "players": [PLAYER_ID["Tal"], PLAYER_ID["Petrosian"]]}).json()
    assert [p["id"] for p in res["players"]] == [PLAYER_ID["Tal"], PLAYER_ID["Petrosian"]]
    for player in res["players"]:
        moves = player["moves"]
        assert {m["uci"] for m in moves} == {m.uci() for m in chess.Board().legal_moves}
        assert sum(m["prob"] for m in moves) == pytest.approx(1, abs=1e-4)
        assert [m["prob"] for m in moves] == sorted((m["prob"] for m in moves), reverse=True)
        assert "e4" in {m["san"] for m in moves}


def test_black_to_move_and_promotions(client):
    board = chess.Board()
    board.push_uci("e2e4")
    res = client.post("/api/predict", json={"fen": board.fen(), "players": [0]}).json()
    assert {m["uci"] for m in res["players"][0]["moves"]} == {m.uci() for m in board.legal_moves}
    res = client.post("/api/predict", json={"fen": PROMOTION, "players": [0]}).json()
    assert {"a7a8q", "a7a8n", "a7a8b", "a7a8r"} <= {m["uci"] for m in res["players"][0]["moves"]}


def test_game_over_has_no_moves(client):
    mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert client.post("/api/predict", json={"fen": mate, "players": [1]}).json()["players"][0]["moves"] == []


def test_bad_requests(client):
    assert client.post("/api/predict", json={"fen": "pas une fen", "players": [1]}).status_code == 400
    assert client.post("/api/predict", json={"fen": chess.Board().fen(), "players": [99]}).status_code == 400
    assert client.post("/api/predict", json={"fen": chess.Board().fen(), "players": []}).status_code == 422


def test_loads_a_run_directory_and_torchscript(tmp_path):
    model = ChessNet(channels=16, blocks=1)
    save(model, {"model": {"channels": 16, "blocks": 1}}, str(tmp_path / "run"))
    run = Predictor(str(tmp_path / "run"))
    assert run.trained and run.name == "run"
    logits = model.eval()(torch.from_numpy(run.game.encode(chess.Board()))[None], torch.tensor([1]))[0]
    logits[~torch.from_numpy(run.game.legal_mask(chess.Board()))] = float("-inf")
    best = logits.argmax().item()
    assert run.predict(chess.Board().fen(), [1])[0][0]["uci"] == run.game.index_to_move(best, chess.Board()).uci()

    export_dummy(str(tmp_path / "dummy.pt"))
    dummy = Predictor(str(tmp_path / "dummy.pt"))
    assert not dummy.trained and len(dummy.predict(chess.Board().fen(), [0, 1])) == 2


@pytest.fixture(scope="module")
def evaluator():
    from chessstyle.evaluation import Evaluator, find_stockfish

    path = find_stockfish()
    if not path:
        pytest.skip("Stockfish non installé")
    ev = Evaluator(path, time_limit=0.1)
    yield ev
    ev.close()


def test_eval_without_stockfish_is_503(client):
    assert client.get("/api/info").json()["engine"] is None
    assert client.post("/api/eval", json={"fen": chess.Board().fen()}).status_code == 503


def test_eval_from_whites_point_of_view(evaluator):
    c = TestClient(create_app(Predictor(), web_dist=None, evaluator=evaluator))
    assert c.get("/api/info").json()["engine"].startswith("Stockfish")
    start = c.post("/api/eval", json={"fen": chess.Board().fen()}).json()
    assert abs(start["cp"]) < 80 and start["best"] in {m.uci() for m in chess.Board().legal_moves}
    # Les Noirs matent en 1 (…Qh4#) : score négatif
    black_mates = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
    assert c.post("/api/eval", json={"fen": black_mates}).json()["mate"] == -1
    # Les Blancs ont une dame de plus : gros avantage ou mat forcé pour eux
    queen_up = c.post("/api/eval", json={"fen": "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"}).json()
    assert (queen_up["mate"] or 0) > 0 or queen_up["cp"] > 500
    mated = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert c.post("/api/eval", json={"fen": mated}).json() == {"cp": None, "mate": 0, "winner": "black"}
    assert c.post("/api/eval", json={"fen": "pas une fen"}).status_code == 400
