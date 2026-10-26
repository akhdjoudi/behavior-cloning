"""Interface générique d'un jeu, pour pouvoir ajouter le go plus tard sans toucher au reste."""

from typing import Any, Protocol

import numpy as np


class Game(Protocol):
    board_shape: tuple[int, int, int]
    num_actions: int

    def encode(self, state: Any) -> np.ndarray:
        """Position -> tenseur uint8 de forme `board_shape`, vu du joueur qui a le trait."""
        ...

    def move_to_index(self, move: Any, state: Any) -> int:
        """Coup légal -> classe dans [0, num_actions)."""
        ...

    def index_to_move(self, idx: int, state: Any) -> Any:
        """Classe -> coup, inverse exact de `move_to_index`."""
        ...

    def legal_mask(self, state: Any) -> np.ndarray:
        """Masque booléen de forme (num_actions,) des coups légaux."""
        ...
