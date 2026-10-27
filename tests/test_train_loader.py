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


def test_balanced_indices_equal_per_player_and_lichess_share():
    from chessstyle.train import balanced_indices

    pid = torch.tensor([0] * 500 + [1] * 300 + [2] * 20)  # le joueur 2 est très sous-représenté
    order = balanced_indices(pid, per_player=50, mix_share=0.2, seed=0)
    counts = torch.bincount(pid[order], minlength=3).tolist()
    assert counts == [25, 50, 50]  # 20 % de Lichess = 25 sur 125
    assert not torch.equal(order, balanced_indices(pid, 50, 0.2, seed=1))  # autre tirage à l'époque suivante


def test_train_with_sampler_and_patience(tmp_path):
    from chessstyle.train import balanced_indices

    data = random_positions(200)
    data["player_id"] = torch.arange(200) % 3
    sampler = lambda e: balanced_indices(data["player_id"], 30, seed=e)
    train(data, str(tmp_path), epochs=50, batch_size=50, val_data=data, sampler=sampler, patience=1,
          model_kwargs={"channels": 16, "blocks": 1})
    assert (tmp_path / "model.pt").exists()
