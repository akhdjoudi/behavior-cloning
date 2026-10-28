"""Pré-entraînement et fine-tune : masque des coups illégaux, mixed precision, logs W&B, checkpoints.

Responsable : rôle 1. Voir la section correspondante du plan de projet.

Format HDF5 attendu (rôle 2) : `boards` (N, 18, 8, 8) uint8, `legal_mask` (N, 4672) bool,
`target` (N,) int, et optionnellement `player_id` (N,) int (0 si absent).
`train` et `evaluate` prennent soit un DataLoader (`dataset.make_loader`, lecture par morceaux),
soit un dict de tenseurs en mémoire (`load_h5`, test de sanité).

    python -m chessstyle.train data/sample/train.h5 --out runs/sanity --epochs 50 --subset 1000
    python -m chessstyle.train data/processed/lichess/train.h5 --val data/processed/lichess/val.h5 --workers 2
"""

import argparse
import json
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
import yaml

from chessstyle.dataset import make_loader
from chessstyle.encoding import ENCODING_VERSION
from chessstyle.model import ChessNet
from chessstyle.players import PLAYERS

MASK_VALUE = -1e9


def load_h5(path: str, subset: int | None = None) -> dict[str, torch.Tensor]:
    with h5py.File(path) as f:
        n = len(f["target"]) if subset is None else min(subset, len(f["target"]))
        data = {
            "board": torch.from_numpy(f["boards"][:n]),
            "mask": torch.from_numpy(f["legal_mask"][:n]).bool(),
            "target": torch.from_numpy(f["target"][:n]).long(),
        }
        data["player_id"] = torch.from_numpy(f["player_id"][:n]).long() if "player_id" in f else torch.zeros(n, dtype=torch.long)
    return data


def balanced_indices(player_id: torch.Tensor, per_player: int, mix_share: float = 0.0, seed: int = 0) -> torch.Tensor:
    """Positions d'une époque : `per_player` par joueur (id >= 1), plus des positions Lichess (id 0) pour `mix_share` de l'époque.

    Les joueurs peu fournis sont tirés avec remise, les autres sans.
    """
    g = torch.Generator().manual_seed(seed)

    def draw(ids: torch.Tensor, k: int) -> torch.Tensor:
        if len(ids) >= k:
            return ids[torch.randperm(len(ids), generator=g)[:k]]
        return ids[torch.randint(len(ids), (k,), generator=g)]

    parts = [draw((player_id == p).nonzero().squeeze(1), per_player) for p in player_id.unique().tolist() if p != 0]
    if mix_share > 0:
        n_mix = round(per_player * len(parts) * mix_share / (1 - mix_share))
        parts.append(draw((player_id == 0).nonzero().squeeze(1), n_mix))
    order = torch.cat(parts)
    return order[torch.randperm(len(order), generator=g)]


def batches(data, batch_size: int, shuffle: bool, order: torch.Tensor | None = None):
    """Itère sur des batchs : un DataLoader est déjà découpé, un dict de tenseurs se découpe ici."""
    if not isinstance(data, dict):
        yield from data
        return
    n = len(data["target"]) if order is None else len(order)
    if order is None:
        order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        idx = order[i : i + batch_size]
        yield {k: v[idx] for k, v in data.items()}


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~mask, MASK_VALUE)


@torch.no_grad()
def evaluate(model: ChessNet, data, device: str, batch_size: int = 256) -> tuple[float, float]:
    """Retourne (perte, top-1) avec le masque des coups illégaux. Un DataLoader doit être créé avec shuffle=False."""
    model.eval()
    loss = correct = n = 0
    for b in batches(data, batch_size, shuffle=False):
        b = {k: v.to(device) for k, v in b.items()}
        n += len(b["target"])
        out = masked_logits(_policy(model(b["board"], b["player_id"])), b["mask"])
        loss += F.cross_entropy(out, b["target"], reduction="sum").item()
        correct += (out.argmax(1) == b["target"]).sum().item()
    return loss / n, correct / n


def _policy(out):
    return out[0] if isinstance(out, tuple) else out


def train(
    data,
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    model_kwargs: dict | None = None,
    val_data=None,
    wandb_project: str | None = None,
    seed: int = 0,
    init: str | None = None,
    sampler=None,
    patience: int | None = None,
) -> ChessNet:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    model_kwargs = {"num_players": len(PLAYERS), **(model_kwargs or {})}
    model = ChessNet(**model_kwargs).to(device)
    if init:  # fine-tune : on repart d'un run précédent (dossier ou model.pt)
        weights = Path(init) / "model.pt" if Path(init).is_dir() else Path(init)
        model.load_state_dict(torch.load(weights, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)  # le surapprentissage arrive vite avec un lr constant
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    config = {
        "model": model_kwargs,
        "train": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay, "seed": seed},
        "encoding_version": ENCODING_VERSION,
    }
    run = None
    if wandb_project:
        import wandb

        run = wandb.init(project=wandb_project, config=config)

    best_top1, stale = -1.0, 0
    for epoch in range(epochs):
        model.train()
        if hasattr(data, "dataset"):
            data.dataset.set_epoch(epoch)  # autre tirage du mélange à chaque époque
        total = n = 0
        order = sampler(epoch) if sampler else None  # tirage équilibré propre à l'époque
        for b in batches(data, batch_size, shuffle=True, order=order):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                logits = _policy(model(b["board"], b["player_id"]))
            loss = F.cross_entropy(masked_logits(logits, b["mask"]), b["target"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(b["target"])
            n += len(b["target"])

        sched.step()
        log = {"epoch": epoch, "train_loss": total / n}
        if val_data is not None:
            log["val_loss"], log["val_top1"] = evaluate(model, val_data, device)
            stale += 1
            if log["val_top1"] > best_top1:  # on garde le meilleur modèle, pas le dernier
                best_top1, stale = log["val_top1"], 0
                save(model, {**config, "best": {"epoch": epoch, "val_top1": best_top1, "val_loss": log["val_loss"]}}, out_dir)
        print(log)
        if run:
            run.log(log)
        if patience and stale >= patience:
            break

    if val_data is None:
        save(model, config, out_dir)
    else:
        model.load_state_dict(torch.load(Path(out_dir) / "model.pt", map_location=device))
    if run:
        run.finish()
    return model


def save(model: ChessNet, config: dict, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (out / "players.json").write_text(json.dumps(PLAYERS, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("h5")
    p.add_argument("--val", help="HDF5 de validation")
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--init", help="run (dossier ou model.pt) dont on repart, pour le fine-tune")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--balance", type=int, metavar="N", help="N positions par joueur à chaque époque (charge le fichier en mémoire)")
    p.add_argument("--mix-h5", help="HDF5 Lichess dont on mélange des positions (id 0) avec --balance")
    p.add_argument("--mix-share", type=float, default=0.2, help="part de Lichess dans une époque")
    p.add_argument("--mix-pool", type=int, default=150_000, help="positions Lichess chargées en mémoire")
    p.add_argument("--patience", type=int, help="arrêt après N époques sans progrès du top-1 de validation")
    p.add_argument("--subset", type=int, help="ne garde que les N premières positions, en mémoire (test de sanité)")
    p.add_argument("--workers", type=int, default=0, help="processus de lecture du DataLoader")
    p.add_argument("--wandb", metavar="PROJET")
    a = p.parse_args()

    sampler = None
    if a.subset:
        data = load_h5(a.h5, a.subset)
        val = load_h5(a.val) if a.val else data  # sans fichier de validation, on mesure sur le train (test de sanité)
    else:
        if a.balance:
            data = load_h5(a.h5)
            if a.mix_h5:
                mix = load_h5(a.mix_h5, a.mix_pool)
                data = {k: torch.cat([data[k], mix[k]]) for k in data}
            sampler = lambda epoch: balanced_indices(data["player_id"], a.balance, a.mix_share if a.mix_h5 else 0.0, seed=epoch)
        else:
            data = make_loader(a.h5, a.batch_size, num_workers=a.workers, drop_last=True)
        val = make_loader(a.val, 256, shuffle=False, num_workers=a.workers) if a.val else None
    train(
        data, a.out, a.epochs, a.batch_size, a.lr, a.weight_decay,
        val_data=val, wandb_project=a.wandb, init=a.init, sampler=sampler, patience=a.patience,
    )


if __name__ == "__main__":
    main()
