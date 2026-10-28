"""Top-k par phase de jeu et par joueur, matrice croisée N x N.

Responsable : rôle 2. Voir la section correspondante du plan de projet.

Chaque position d'un HDF5 de `build_dataset.py` passe dans le modèle conditionné tour à tour sur
chaque joueur de sa liste. On en tire :
- pour chaque joueur j : top-1, top-3 et perte (log-vraisemblance négative du coup joué) du modèle
  conditionné sur j, sur les positions de j, au total et par phase de jeu ;
- la matrice croisée : case (i, j) = modèle conditionné sur i, évalué sur les vrais coups de j.
  Si le modèle a appris les styles, la diagonale domine chaque colonne.

Les coups 1 à 10 (demi-coups 0 à 19) relèvent surtout de la théorie d'ouverture : ils forment la phase
« ouverture », mesurée à part et exclue des chiffres globaux et de la matrice. La finale commence quand
il reste au plus 6 dames, tours, fous et cavaliers sur l'échiquier (critère du découpage de Lichess).

    python -m chessstyle.evaluate runs/latest data/processed/icons/test.h5 --out runs/latest/eval_icons.json
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from chessstyle.build_dataset import CHUNK
from chessstyle.encoding import ENCODING_VERSION
from chessstyle.model import ChessNet
from chessstyle.players import GENERIC, PLAYERS

PHASES = ("ouverture", "milieu", "finale")
OPENING_PLIES = 20  # coups 1 à 10
ENDGAME_PIECES = 6
PIECE_PLANES = [1, 2, 3, 4, 7, 8, 9, 10]  # cavaliers, fous, tours et dames des deux camps


def phase_of(boards: np.ndarray, ply: np.ndarray) -> np.ndarray:
    """Indice dans PHASES de chaque position."""
    pieces = boards[:, PIECE_PLANES].reshape(len(boards), -1).sum(1, dtype=np.int64)
    return np.where(ply < OPENING_PLIES, 0, np.where(pieces <= ENDGAME_PIECES, 2, 1))


def load_model(path: str) -> tuple[torch.nn.Module, list[str]]:
    """Dossier de run de `train.py` (model.pt, config.yaml, players.json) ou modèle TorchScript."""
    p = Path(path)
    if not p.is_dir():
        return torch.jit.load(p, map_location="cpu").eval(), list(PLAYERS)
    config = yaml.safe_load((p / "config.yaml").read_text())
    if config.get("encoding_version") != ENCODING_VERSION:
        raise ValueError(f"{p} : encodage v{config.get('encoding_version')}, ce code lit la v{ENCODING_VERSION}")
    model = ChessNet(**config["model"])
    model.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
    players = json.loads((p / "players.json").read_text()) if (p / "players.json").exists() else list(PLAYERS)
    return model.eval(), players


@torch.no_grad()
def score(
    model: torch.nn.Module, path: str, players: list[str], batch_size: int = 256, device: str = "cpu", limit: int | None = None
) -> dict[str, np.ndarray]:
    """Par position : joueur réel (indice dans `players`, -1 s'il n'y est pas), phase, et pour chaque
    joueur i de `players` le rang du coup joué (0 = le plus probable) et sa perte, en tableaux (i, position)."""
    model.to(device)
    num = len(players)
    out = {"player": [], "phase": [], "rank": [], "nll": []}
    with h5py.File(path) as f:
        version = f.attrs.get("encoding_version", ENCODING_VERSION)
        if version != ENCODING_VERSION:
            raise ValueError(f"{path} : encodage v{version}, ce code lit la v{ENCODING_VERSION}")
        names = json.loads(f.attrs["players"]) if "players" in f.attrs else list(PLAYERS)
        to_model = np.array([players.index(n) if n in players else -1 for n in names])
        n = len(f["target"]) if limit is None else min(limit, len(f["target"]))

        for start in range(0, n, CHUNK):  # un chunk HDF5 à la fois, il n'est décompressé qu'une fois
            stop = min(start + CHUNK, n)
            boards = f["boards"][start:stop]
            masks = f["legal_mask"][start:stop]
            targets = f["target"][start:stop].astype(np.int64)
            out["player"].append(to_model[f["player_id"][start:stop]])
            out["phase"].append(phase_of(boards, f["ply"][start:stop]))

            for b in range(0, len(targets), batch_size):
                k = len(targets[b : b + batch_size])
                x = torch.from_numpy(boards[b : b + batch_size]).to(device).repeat(num, 1, 1, 1)
                ids = torch.arange(num, device=device).repeat_interleave(k)  # joueur 0 sur tout le batch, puis 1…
                logits = model(x, ids)
                logits = (logits[0] if isinstance(logits, tuple) else logits).float().view(num, k, -1)
                mask = torch.from_numpy(masks[b : b + batch_size]).to(device)
                logp = logits.masked_fill(~mask, float("-inf")).log_softmax(-1)
                target = torch.from_numpy(targets[b : b + batch_size]).to(device)
                played = logp.gather(-1, target.expand(num, k)[..., None])
                out["rank"].append((logp > played).sum(-1).cpu().numpy())
                out["nll"].append(-played.squeeze(-1).cpu().numpy())

    return {
        "player": np.concatenate(out["player"]),
        "phase": np.concatenate(out["phase"]),
        "rank": np.concatenate(out["rank"], axis=1),
        "nll": np.concatenate(out["nll"], axis=1),
    }


def _metrics(s: dict[str, np.ndarray], rows: np.ndarray, i: int) -> dict:
    """Top-1, top-3 et perte du modèle conditionné sur i, sur les positions `rows`."""
    if not rows.any():
        return {"positions": 0, "top1": None, "top3": None, "nll": None}
    rank = s["rank"][i, rows]
    return {
        "positions": int(rows.sum()),
        "top1": float((rank == 0).mean()),
        "top3": float((rank < 3).mean()),
        "nll": float(s["nll"][i, rows].mean()),
    }


def summarize(s: dict[str, np.ndarray], players: list[str]) -> dict:
    """Chiffres par joueur et par phase, et matrice croisée (hors ouverture)."""
    played = s["phase"] > 0
    present = [j for j in range(len(players)) if (s["player"] == j).any()]
    generic = players.index(PLAYERS[GENERIC]) if PLAYERS[GENERIC] in players else None

    by_player = {}
    for j in present:
        own = s["player"] == j
        entry = _metrics(s, own & played, j)
        entry["phases"] = {name: _metrics(s, own & (s["phase"] == p), j) for p, name in enumerate(PHASES)}
        if generic is not None and j != generic and entry["top1"] is not None:
            entry["gain_vs_generique"] = entry["top1"] - _metrics(s, own & played, generic)["top1"]
        by_player[players[j]] = entry

    top1 = [[_metrics(s, (s["player"] == j) & played, i)["top1"] for j in present] for i in range(len(players))]
    nll = [[_metrics(s, (s["player"] == j) & played, i)["nll"] for j in present] for i in range(len(players))]
    cols = [players[j] for j in present]

    def diagonal_wins(matrix: list[list[float | None]], best) -> int:
        """Colonnes où le modèle conditionné sur le bon joueur fait au moins aussi bien que tous les autres."""
        wins = 0
        for c, j in enumerate(present):
            column = [row[c] for row in matrix if row[c] is not None]
            wins += bool(column) and matrix[j][c] == best(column)
        return wins

    everyone = (s["player"] >= 0) & played
    true_player = np.where(s["player"] >= 0, s["player"], 0)
    own_rank = s["rank"][true_player, np.arange(len(true_player))]
    own_nll = s["nll"][true_player, np.arange(len(true_player))]
    overall = {"positions": int(everyone.sum())}
    if everyone.any():
        overall |= {
            "top1": float((own_rank[everyone] == 0).mean()),
            "top3": float((own_rank[everyone] < 3).mean()),
            "nll": float(own_nll[everyone].mean()),
        }

    return {
        "overall": overall,
        "players": by_player,
        "cross": {
            "rows": list(players),
            "cols": cols,
            "top1": top1,
            "nll": nll,
            "diagonal_wins_top1": diagonal_wins(top1, max),
            "diagonal_wins_nll": diagonal_wins(nll, min),
        },
    }


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def report(result: dict) -> str:
    """Tableaux lisibles dans un terminal."""
    lines = []
    o = result["overall"]
    lines.append(f"{o['positions']} positions hors ouverture · top-1 {_pct(o.get('top1'))} % · top-3 {_pct(o.get('top3'))} %")
    lines.append("")
    head = f"{'joueur':<12}{'positions':>10}{'top-1':>8}{'top-3':>8}{'perte':>8}" + "".join(f"{p:>11}" for p in PHASES)
    lines.append(head + f"{'vs générique':>14}")
    for name, e in result["players"].items():
        phases = "".join(f"{_pct(e['phases'][p]['top1']):>11}" for p in PHASES)
        gain = e.get("gain_vs_generique")
        gain = "" if gain is None else f"{100 * gain:+.1f}"
        nll = "—" if e["nll"] is None else f"{e['nll']:.2f}"
        lines.append(f"{name:<12}{e['positions']:>10}{_pct(e['top1']):>8}{_pct(e['top3']):>8}{nll:>8}{phases}{gain:>14}")

    c = result["cross"]
    for key, title, best in (("top1", "top-1 (%)", max), ("nll", "perte", min)):
        lines += ["", f"Matrice croisée, {title} : ligne = modèle conditionné sur, colonne = vrais coups de ; * = meilleur de la colonne"]
        lines.append(f"{'':<12}" + "".join(f"{col[:10]:>12}" for col in c["cols"]))
        for i, row in enumerate(c[key]):
            cells = []
            for col, value in enumerate(row):
                column = [r[col] for r in c[key] if r[col] is not None]
                text = "—" if value is None else (_pct(value) if key == "top1" else f"{value:.3f}")
                mark = "*" if value is not None and value == best(column) else " "
                cells.append(f"{text}{mark}".rjust(12))
            lines.append(f"{c['rows'][i]:<12}" + "".join(cells))
    lines += ["", f"Diagonale meilleure de sa colonne : {c['diagonal_wins_top1']}/{len(c['cols'])} en top-1, "
              f"{c['diagonal_wins_nll']}/{len(c['cols'])} en perte"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="dossier de run de train.py, ou modèle TorchScript")
    p.add_argument("h5", help="HDF5 de build_dataset.py, en général le split test")
    p.add_argument("--out", help="fichier JSON des résultats")
    p.add_argument("--batch-size", type=int, default=256, help="positions par passage, multipliées par le nombre de joueurs")
    p.add_argument("--limit", type=int, help="ne garde que les N premières positions")
    p.add_argument("--wandb", metavar="PROJET")
    a = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, players = load_model(a.model)
    result = summarize(score(model, a.h5, players, a.batch_size, device, a.limit), players)
    result["model"], result["data"] = str(a.model), str(a.h5)
    print(report(result))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if a.wandb:
        import wandb

        c = result["cross"]
        run = wandb.init(project=a.wandb, job_type="eval", config={"model": a.model, "data": a.h5})
        log = {f"eval/{k}": v for k, v in result["overall"].items()}
        for name, e in result["players"].items():
            log |= {f"eval/{name}/{k}": e[k] for k in ("top1", "top3", "nll")}
            log |= {f"eval/{name}/top1_{phase}": e["phases"][phase]["top1"] for phase in PHASES}
        log["eval/cross_top1"] = wandb.Table(columns=["modele", *c["cols"]], data=[[r, *v] for r, v in zip(c["rows"], c["top1"])])
        run.log(log)
        run.finish()
    return result


if __name__ == "__main__":
    main()
