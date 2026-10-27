"""Inférence pour l'appli : distribution des coups légaux d'une position, pour un ou plusieurs joueurs.

Responsable : rôle 2. Voir la section correspondante du plan de projet.

Charge au choix :
- un dossier de run de `train.py` (model.pt = state_dict, config.yaml, players.json) ;
- un modèle TorchScript (`dummy_model.pt` de `model.export_dummy`) ;
- rien : modèle aux poids aléatoires, pour développer l'interface.
"""

import json
from pathlib import Path

import chess
import numpy as np
import torch
import yaml

from chessstyle.encoding import ChessGame
from chessstyle.model import ChessNet
from chessstyle.players import PLAYERS


class Predictor:
    def __init__(self, path: str | None = None):
        self.game = ChessGame()
        self.players = list(PLAYERS)
        self.trained = path is not None
        self.name = "aléatoire"
        if path is None:
            self.model = ChessNet()
        elif Path(path).is_dir():
            run = Path(path)
            config = yaml.safe_load((run / "config.yaml").read_text())
            self.model = ChessNet(**config["model"])
            self.model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
            if (run / "players.json").exists():
                self.players = json.loads((run / "players.json").read_text())
            self.name = run.name
        else:
            self.model = torch.jit.load(path, map_location="cpu")
            self.trained = False  # export_dummy : poids aléatoires
            self.name = Path(path).name
        self.model.eval()

    @torch.no_grad()
    def predict(self, fen: str, player_ids: list[int]) -> list[list[dict]]:
        """Pour chaque joueur, les coups légaux triés par probabilité décroissante : uci, san, prob."""
        board = chess.Board(fen)
        legal = self.game.legal_mask(board)
        if not legal.any():
            return [[] for _ in player_ids]
        x = torch.from_numpy(np.repeat(self.game.encode(board)[None], len(player_ids), axis=0))
        out = self.model(x, torch.tensor(player_ids, dtype=torch.long))
        logits = (out[0] if isinstance(out, tuple) else out).float()
        logits[:, ~torch.from_numpy(legal)] = float("-inf")
        probs = logits.softmax(1)

        indices = np.flatnonzero(legal)
        moves = [self.game.index_to_move(int(i), board) for i in indices]
        sans = [board.san(m) for m in moves]
        result = []
        for row in probs[:, indices].tolist():
            ranked = sorted(zip(moves, sans, row), key=lambda t: -t[2])
            result.append([{"uci": m.uci(), "san": s, "prob": p} for m, s, p in ranked])
        return result
