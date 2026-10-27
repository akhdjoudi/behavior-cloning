"""API de l'appli de démo, et service des fichiers de l'interface React une fois compilée.

Responsable : rôle 2. Voir la section correspondante du plan de projet.

    python -m chessstyle.server --model runs/latest          # dossier de run de train.py
    python -m chessstyle.server --model dummy_model.pt       # TorchScript aux poids aléatoires
    python -m chessstyle.server                              # sans modèle : poids aléatoires

GET  /api/info     joueurs du modèle, modèle entraîné ou non, moteur d'évaluation
POST /api/predict  {"fen": ..., "players": [1, 2]} -> coups légaux triés par probabilité, par joueur
POST /api/eval     {"fen": ...} -> évaluation Stockfish du point de vue des Blancs (503 sans Stockfish)
"""

import argparse
import os
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chessstyle.encoding import ENCODING_VERSION
from chessstyle.evaluation import Evaluator, find_stockfish
from chessstyle.inference import Predictor

WEB_DIST = Path(__file__).resolve().parents[2] / "app" / "dist"


class PredictRequest(BaseModel):
    fen: str
    players: list[int] = Field(min_length=1, max_length=32)


class EvalRequest(BaseModel):
    fen: str


def _check_fen(fen: str) -> None:
    try:
        chess.Board(fen)
    except ValueError as e:
        raise HTTPException(400, f"FEN invalide : {e}") from e


def create_app(predictor: Predictor, web_dist: Path | None = WEB_DIST, evaluator: Evaluator | None = None) -> FastAPI:
    app = FastAPI(title="Chess Style Cloning")

    @app.get("/api/info")
    def info():
        return {
            "players": predictor.players,
            "trained": predictor.trained,
            "model": predictor.name,
            "encoding_version": ENCODING_VERSION,
            "engine": evaluator.name if evaluator else None,
        }

    @app.post("/api/predict")
    def predict(req: PredictRequest):
        _check_fen(req.fen)
        if any(not 0 <= p < len(predictor.players) for p in req.players):
            raise HTTPException(400, f"player_id hors de 0..{len(predictor.players) - 1}")
        moves = predictor.predict(req.fen, req.players)
        return {"players": [{"id": p, "moves": m} for p, m in zip(req.players, moves)]}

    @app.post("/api/eval")
    def evaluate(req: EvalRequest):
        if evaluator is None:
            raise HTTPException(503, "Stockfish non disponible")
        _check_fen(req.fen)
        return evaluator.evaluate(req.fen)

    # Après les routes /api : l'interface compilée (npm run build) est servie à la racine
    if web_dist is not None and web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app


if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL_PATH"), help="dossier de run ou fichier TorchScript")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    p.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH") or find_stockfish(), help="binaire Stockfish (cherché dans le PATH)")
    p.add_argument("--no-eval", action="store_true", help="désactiver la barre d'évaluation")
    args = p.parse_args()
    evaluator = None if args.no_eval or not args.stockfish else Evaluator(args.stockfish)
    print(f"Évaluation : {evaluator.name if evaluator else 'désactivée (Stockfish introuvable)'}")
    try:
        uvicorn.run(create_app(Predictor(args.model), evaluator=evaluator), host=args.host, port=args.port)
    finally:
        if evaluator:
            evaluator.close()
