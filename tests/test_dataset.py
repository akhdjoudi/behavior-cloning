import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chessstyle.build_dataset import CHUNK, FIELDS
from chessstyle.dataset import make_loader

N = 3 * CHUNK + 1000  # 4 chunks, le dernier incomplet


@pytest.fixture(scope="module")
def h5(tmp_path_factory):
    """Fichier au format de build_dataset où le numéro de ligne est caché dans les deux premières cases du plateau."""
    path = tmp_path_factory.mktemp("data") / "train.h5"
    rng = np.random.default_rng(0)
    rows = np.arange(N)
    boards = rng.integers(0, 2, (N, 18, 8, 8), dtype=np.uint8)
    boards[:, 0, 0, 0], boards[:, 0, 0, 1] = rows % 256, rows // 256
    data = {
        "boards": boards,
        "legal_mask": rng.random((N, 4672)) < 0.01,
        "target": rng.integers(0, 4672, N),
        "player_id": rng.integers(0, 7, N),
    }
    with h5py.File(path, "w") as f:
        for k, (shape, dtype) in FIELDS.items():
            f.create_dataset(k, shape=(N, *shape), dtype=dtype, chunks=(CHUNK, *shape), compression="gzip")
            if k in data:
                f[k][:] = data[k]
    return str(path), data


def row_ids(batch) -> np.ndarray:
    board = batch["board"].numpy()
    return board[:, 0, 0, 0].astype(np.int64) + 256 * board[:, 0, 0, 1].astype(np.int64)


def served(loader) -> np.ndarray:
    return np.concatenate([row_ids(b) for b in loader])


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("shuffle", [False, True])
def test_every_position_once_per_epoch(h5, shuffle, num_workers):
    rows = served(make_loader(h5[0], batch_size=500, shuffle=shuffle, num_workers=num_workers))
    assert np.array_equal(np.sort(rows), np.arange(N))


def test_rows_stay_aligned_after_shuffle(h5):
    path, data = h5
    for batch in make_loader(path, batch_size=500, shuffle=True, num_workers=2):
        rows = row_ids(batch)
        assert np.array_equal(batch["target"].numpy(), data["target"][rows])
        assert np.array_equal(batch["player_id"].numpy(), data["player_id"][rows])
        assert np.array_equal(batch["mask"].numpy(), data["legal_mask"][rows])


def test_batches_are_full_except_the_last(h5):
    sizes = [len(b["target"]) for b in make_loader(h5[0], batch_size=500, buffer_chunks=2)]
    assert all(s == 500 for s in sizes[:-1]) and 0 < sizes[-1] <= 500 and sum(sizes) == N
    dropped = [len(b["target"]) for b in make_loader(h5[0], batch_size=500, buffer_chunks=2, drop_last=True)]
    assert set(dropped) == {500}


def test_batch_keys_and_types_match_train_load_h5(h5):
    batch = next(iter(make_loader(h5[0], batch_size=64)))
    assert batch["board"].shape == (64, 18, 8, 8) and batch["board"].dtype == torch.uint8
    assert batch["mask"].shape == (64, 4672) and batch["mask"].dtype == torch.bool
    assert batch["target"].dtype == batch["player_id"].dtype == torch.int64


def test_shuffle_depends_on_epoch_and_is_reproducible(h5):
    loader = make_loader(h5[0], batch_size=500, seed=1)
    epoch0 = served(loader)
    loader.dataset.set_epoch(1)
    epoch1 = served(loader)
    loader.dataset.set_epoch(0)
    assert np.array_equal(served(loader), epoch0)
    assert not np.array_equal(epoch0, epoch1)
    assert not np.array_equal(epoch0, np.arange(N))


def test_a_batch_goes_through_the_model(h5):
    from chessstyle.model import ChessNet
    from chessstyle.train import masked_logits

    batch = next(iter(make_loader(h5[0], batch_size=32)))
    batch["mask"][range(32), batch["target"]] = True  # dans les vraies données, le coup joué est toujours légal
    model = ChessNet(channels=16, blocks=1)
    logits = masked_logits(model(batch["board"], batch["player_id"]), batch["mask"])
    loss = torch.nn.functional.cross_entropy(logits, batch["target"])
    loss.backward()
    assert torch.isfinite(loss)
