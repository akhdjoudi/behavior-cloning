# Chess Style Cloning

Un bot d'échecs qui imite, au choix, le style de plusieurs joueurs iconiques (Tal, Petrossian, Carlsen…), par behavioral cloning avec PyTorch.
Livrable : une démo jouable avec les probabilités de coups affichées sur l'échiquier, et un benchmark qui prouve la différence de style.

Plan de projet complet (vision, tâches par rôle, planning) : https://claude.ai/artifact/VpVWEHbMFxdTUavUfF3bGG

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install stockfish
wandb login
pytest -q
```

Sous Windows : `.venv\Scriptsctivate` à la place de `source`, et Stockfish via `winget install Stockfish.Stockfish` (ou le binaire de stockfishchess.org, ajouté au `PATH`).

## Structure

```
src/chessstyle/
├── game.py           # interface Game générique
├── encoding.py       # ChessGame : le contrat (voir docs/CONTRACT.md)
├── players.py        # liste figée des joueurs imités
├── ingest.py         # rôle 1 : PGN -> parties filtrées
├── build_dataset.py  # rôle 2 : parties -> HDF5
├── dataset.py        # rôle 2 : DataLoader
├── model.py          # rôle 1 : ResNet conditionné par le joueur
├── train.py          # rôle 1 : pré-entraînement et fine-tune
├── evaluate.py       # rôle 2 : top-k, matrice croisée
└── style_bench.py    # rôle 2 : métriques de style Stockfish
configs/   app/   tests/   docs/   data/ (ignoré par Git)
```

## Règles

- `main` ne reçoit que des PR relues par l'autre, avec la CI verte.
- Une branche par issue : `feat/...`, `fix/...`.
- `encoding.py` ne change qu'avec l'accord des deux et une nouvelle `ENCODING_VERSION`.
- Les décisions vont dans `docs/DECISIONS.md`.
