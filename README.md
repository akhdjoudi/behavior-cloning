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

Sous Windows : `.venv\Scripts\activate` à la place de `source`, et Stockfish via `winget install Stockfish.Stockfish` (ou le binaire de stockfishchess.org, ajouté au `PATH`).

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
├── style_bench.py    # rôle 2 : métriques de style Stockfish
├── inference.py      # rôle 2 : distribution des coups d'une position, par joueur
└── server.py         # rôle 2 : API FastAPI de l'appli
app/       # rôle 2 : interface React (chessground), modes Analyse, Jouer, Duel
configs/   tests/   docs/   data/ et runs/ (ignorés par Git)
```

## Appli de démo

Il faut Node.js 22 ou plus récent.

```bash
cd app && npm ci && npm run build && cd ..
python -m chessstyle.server --model runs/latest     # http://127.0.0.1:8000
```

`--model` accepte un dossier de run de `train.py` ou un fichier TorchScript (`python -m chessstyle.model` crée `dummy_model.pt`). Sans `--model`, les poids sont aléatoires.
Pour développer l'interface avec rechargement à chaud : lancer le serveur, puis `cd app && npm run dev` (http://localhost:5173, les appels `/api` partent vers le port 8000).

## Règles

- `main` ne reçoit que des PR relues par l'autre, avec la CI verte.
- Une branche par issue : `feat/...`, `fix/...`.
- `encoding.py` ne change qu'avec l'accord des deux et une nouvelle `ENCODING_VERSION`.
- Les décisions vont dans `docs/DECISIONS.md`.
