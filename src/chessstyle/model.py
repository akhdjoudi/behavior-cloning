"""ResNet conditionné par le joueur (embedding + FiLM), tête de politique à 73 canaux.

Responsable : rôle 1. Voir la section correspondante du plan de projet.

Signature : forward(board, player_id) avec board (B, 18, 8, 8) uint8 (contrat d'interface) et
player_id (B,) int64. Retourne les logits (B, 4672), indexés `type * 64 + case_de_départ`,
ou (logits, valeur) si la tête de valeur est activée. Le masque des coups illégaux est appliqué
à l'entraînement et à l'inférence, pas ici.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessstyle.encoding import NUM_MOVE_TYPES, NUM_PLANES, PLANE_HALFMOVE
from chessstyle.players import PLAYERS


class FiLMResBlock(nn.Module):
    """Bloc résiduel dont la branche convolutive est modulée par le joueur : gain * h + biais, par canal."""

    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.film = nn.Linear(emb_dim, 2 * channels)
        # Au départ gain = 1 et biais = 0 : le bloc est un ResNet classique, le joueur s'apprend ensuite
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        delta_gain, bias = self.film(emb)[:, :, None, None].chunk(2, dim=1)
        h = (1 + delta_gain) * h + bias
        return F.relu(x + h)


class ChessNet(nn.Module):
    def __init__(
        self,
        num_players: int = len(PLAYERS),
        channels: int = 128,
        blocks: int = 8,
        emb_dim: int = 64,
        value_head: bool = False,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.embedding = nn.Embedding(num_players, emb_dim)
        self.blocks = nn.ModuleList(FiLMResBlock(channels, emb_dim) for _ in range(blocks))
        self.policy = nn.Conv2d(channels, NUM_MOVE_TYPES, 1)  # 73 canaux x 64 cases = 4 672 logits
        self.value = (
            nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
            if value_head
            else None
        )

    def forward(self, board: torch.Tensor, player_id: torch.Tensor):
        x = board.float()
        # Le plan du compteur de demi-coups va de 0 à 100 : on le ramène dans [0, 1]
        x = torch.cat([x[:, :PLANE_HALFMOVE], x[:, PLANE_HALFMOVE:] / 100.0], dim=1)
        x = self.stem(x)
        emb = self.embedding(player_id)
        for block in self.blocks:
            x = block(x, emb)
        logits = self.policy(x).flatten(1)  # (B, 73, 8, 8) -> (B, 4672), index = type * 64 + case
        if self.value is None:
            return logits
        return logits, self.value(x).squeeze(1)


def export_dummy(path: str = "dummy_model.pt") -> None:
    """Modèle TorchScript aux poids aléatoires, pour que le rôle 2 branche l'appli avant la fin de l'entraînement."""
    model = ChessNet().eval()
    example = (torch.zeros(1, NUM_PLANES, 8, 8, dtype=torch.uint8), torch.zeros(1, dtype=torch.long))
    torch.jit.trace(model, example).save(path)


if __name__ == "__main__":
    export_dummy()
