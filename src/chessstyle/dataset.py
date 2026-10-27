"""Dataset et DataLoader PyTorch : lecture chunk par chunk, mélange dans un buffer.

Responsable : rôle 2. Voir la section correspondante du plan de projet.

Lit les fichiers de `build_dataset.py` sans les charger en mémoire : à chaque époque, l'ordre des
chunks de 4 096 lignes est tiré au hasard, `buffer_chunks` chunks sont lus d'un bloc puis mélangés
ligne à ligne avant d'être découpés en batchs complets (le reste passe au buffer suivant ; seul le
dernier batch d'un worker peut être plus petit, `drop_last=True` l'enlève). Mémoire par worker : environ
`buffer_chunks * 4096 * 5,8 Ko` (190 Mo avec 8 chunks).

Les batchs ont les mêmes clés que `train.load_h5` : `board`, `mask`, `target`, `player_id`.

    loader = make_loader("data/processed/lichess/train.h5", batch_size=512, num_workers=2)
    for epoch in range(epochs):
        loader.dataset.set_epoch(epoch)
        for batch in loader:
            ...
"""

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from chessstyle.build_dataset import CHUNK

FIELDS = ("boards", "legal_mask", "target", "player_id")


class H5Positions(IterableDataset):
    """Positions d'un fichier HDF5, servies par batchs déjà constitués."""

    def __init__(
        self, path: str, batch_size: int, shuffle: bool = True, buffer_chunks: int = 8, seed: int = 0, drop_last: bool = False
    ):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_chunks = buffer_chunks
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        with h5py.File(path) as f:
            self.num_positions = f["target"].shape[0]

    def set_epoch(self, epoch: int) -> None:
        """Change le tirage du mélange ; à appeler avant chaque époque."""
        self.epoch = epoch

    def _chunks(self) -> list[int]:
        """Débuts des chunks attribués à ce worker, dans l'ordre de lecture de l'époque."""
        starts = np.arange(0, self.num_positions, CHUNK)
        if self.shuffle:
            np.random.default_rng((self.seed, self.epoch)).shuffle(starts)
        worker = get_worker_info()
        if worker is not None:
            starts = starts[worker.id :: worker.num_workers]
        return starts.tolist()

    def __iter__(self):
        worker = get_worker_info()
        rng = np.random.default_rng((self.seed, self.epoch, worker.id if worker else 0))
        starts = self._chunks()
        rest = None  # fin du buffer précédent, gardée pour ne produire que des batchs complets
        with h5py.File(self.path) as f:
            for i in range(0, len(starts), self.buffer_chunks):
                # Les chunks sont lus entiers : c'est l'unité de compression, une lecture partielle coûterait autant
                blocks = [{k: f[k][s : s + CHUNK] for k in FIELDS} for s in starts[i : i + self.buffer_chunks]]
                if rest is not None:
                    blocks.append(rest)
                buffer = {k: np.concatenate([b[k] for b in blocks]) for k in FIELDS}
                n = len(buffer["target"])
                order = rng.permutation(n) if self.shuffle else np.arange(n)
                full = n - n % self.batch_size
                for j in range(0, full, self.batch_size):
                    yield _to_torch(buffer, order[j : j + self.batch_size])
                rest = {k: v[order[full:]] for k, v in buffer.items()} if full < n else None
        if rest is not None and not self.drop_last:
            yield _to_torch(rest, np.arange(len(rest["target"])))


def _to_torch(buffer: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "board": torch.from_numpy(buffer["boards"][idx]),
        "mask": torch.from_numpy(buffer["legal_mask"][idx]),
        "target": torch.from_numpy(buffer["target"][idx].astype(np.int64)),
        "player_id": torch.from_numpy(buffer["player_id"][idx].astype(np.int64)),
    }


def make_loader(
    path: str,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    buffer_chunks: int = 8,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    dataset = H5Positions(path, batch_size, shuffle, buffer_chunks, seed, drop_last)
    # batch_size=None : le dataset rend déjà des batchs, le DataLoader ne fait que paralléliser la lecture
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,  # workers recréés à chaque époque : ils voient le set_epoch
    )
