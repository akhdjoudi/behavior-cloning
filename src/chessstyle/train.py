"""Pré-entraînement et fine-tune : masque des coups illégaux, mixed precision, logs W&B, checkpoints.

Responsable : rôle 1. Voir la section correspondante du plan de projet.

Format HDF5 attendu (rôle 2) : `boards` (N, 18, 8, 8) uint8, `legal_mask` (N, 4672) bool,
`target` (N,) int, et optionnellement `player_id` (N,) int (0 si absent).

    python -m chessstyle.train data/sample.h5 --out runs/sanity --epochs 50 --subset 1000
"""

import argparse
import json
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
import yaml

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


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~mask, MASK_VALUE)


@torch.no_grad()
def evaluate(model: ChessNet, data: dict[str, torch.Tensor], device: str, batch_size: int = 256) -> tuple[float, float]:
    """Retourne (perte, top-1) avec le masque des coups illégaux."""
    model.eval()
    loss = correct = 0.0
    n = len(data["target"])
    for i in range(0, n, batch_size):
        b = {k: v[i : i + batch_size].to(device) for k, v in data.items()}
        out = masked_logits(_policy(model(b["board"], b["player_id"])), b["mask"])
        loss += F.cross_entropy(out, b["target"], reduction="sum").item()
        correct += (out.argmax(1) == b["target"]).sum().item()
    return loss / n, correct / n


def _policy(out):
    return out[0] if isinstance(out, tuple) else out


def train(
    data: dict[str, torch.Tensor],
    out_dir: str,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    model_kwargs: dict | None = None,
    val_data: dict[str, torch.Tensor] | None = None,
    wandb_project: str | None = None,
    seed: int = 0,
) -> ChessNet:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    model_kwargs = {"num_players": len(PLAYERS), **(model_kwargs or {})}
    model = ChessNet(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

    n = len(data["target"])
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            b = {k: v[idx].to(device) for k, v in data.items()}
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                logits = _policy(model(b["board"], b["player_id"]))
            loss = F.cross_entropy(masked_logits(logits, b["mask"]), b["target"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(idx)

        log = {"epoch": epoch, "train_loss": total / n}
        if val_data is not None:
            log["val_loss"], log["val_top1"] = evaluate(model, val_data, device)
        print(log)
        if run:
            run.log(log)

    save(model, config, out_dir)
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
    p.add_argument("--subset", type=int, help="ne garde que les N premières positions")
    p.add_argument("--wandb", metavar="PROJET")
    a = p.parse_args()

    data = load_h5(a.h5, a.subset)
    val = load_h5(a.val) if a.val else data  # sans fichier de validation, on mesure sur le train (test de sanité)
    train(
        data, a.out, a.epochs, a.batch_size, a.lr,
        val_data=val, wandb_project=a.wandb,
    )


if __name__ == "__main__":
    main()
