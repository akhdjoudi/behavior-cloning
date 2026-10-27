"""train() et evaluate() avec le DataLoader HDF5 : même chemin que le pré-entraînement sur Lichess."""

import h5py
import pytest
import yaml

torch = pytest.importorskip("torch")

from chessstyle.build_dataset import CHUNK, FIELDS
from chessstyle.dataset import make_loader
from chessstyle.train import evaluate, train
from test_overfit import random_positions


def test_train_and_evaluate_from_loader(tmp_path):
    data = random_positions(300)
    path = tmp_path / "train.h5"
    with h5py.File(path, "w") as f:
        keys = {"boards": "board", "legal_mask": "mask", "target": "target", "player_id": "player_id"}
        for k, (shape, dtype) in FIELDS.items():
            f.create_dataset(k, shape=(300, *shape), dtype=dtype, chunks=(CHUNK, *shape), maxshape=(None, *shape))
            if k in keys:
                f[k][:] = data[keys[k]].numpy()

    train_loader = make_loader(str(path), 100, drop_last=True)
    val_loader = make_loader(str(path), 100, shuffle=False)
    model = train(train_loader, str(tmp_path / "out"), epochs=2, val_data=val_loader, model_kwargs={"channels": 16, "blocks": 1})
    loss, top1 = evaluate(model, val_loader, "cpu")
    assert loss == loss and 0 <= top1 <= 1  # fini, pas NaN
    assert (tmp_path / "out" / "model.pt").exists()


def test_best_checkpoint_and_init(tmp_path):
    """Avec une validation, model.pt est le meilleur modèle ; --init repart d'un run précédent."""
    data = random_positions(200)
    kw = {"channels": 16, "blocks": 1}
    first = train(data, str(tmp_path / "a"), epochs=3, batch_size=100, val_data=data, model_kwargs=kw)
    best = yaml.safe_load((tmp_path / "a" / "config.yaml").read_text())["best"]
    assert 0 <= best["epoch"] < 3
    assert evaluate(first, data, "cpu")[1] == pytest.approx(best["val_top1"])  # le modèle rendu est le meilleur

    second = train(data, str(tmp_path / "b"), epochs=1, batch_size=100, lr=0.0, model_kwargs=kw, init=str(tmp_path / "a"))
    # lr nul : les poids repartis ne bougent pas (les statistiques de BatchNorm, si)
    assert torch.equal(second.stem[0].weight, first.stem[0].weight)
