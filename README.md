# Chess Style Cloning

Un bot d'échecs qui imite, au choix, le style de plusieurs joueurs iconiques (Tal, Petrossian, Carlsen…), par behavioral cloning avec PyTorch.
Livrable : une démo jouable avec les probabilités de coups affichées sur l'échiquier, et un benchmark qui mesure la différence de style.

**Résultat en une phrase** : un ResNet conditionné par un embedding de joueur, pré-entraîné sur Lichess puis ajusté sur 6 grands maîtres, prédit le coup réel avec **39,3 %** de top-1 (hors ouverture) contre 37,4 % pour le modèle Lichess seul. Le conditionnement par joueur n'apporte qu'environ 0,6 point de plus, et le benchmark de style **ne démontre pas** que le modèle distingue les styles (voir [Résultats](#résultats) et [Limites](#limites)).

Plan de projet complet (vision, tâches par rôle, planning) : https://claude.ai/artifact/VpVWEHbMFxdTUavUfF3bGG

## Démo

![Démo : la même position vue par Tal, Petrosian, puis par les six joueurs](docs/img/demo.gif)

Est-indienne, coup 13, trait aux Noirs. Le modèle conditionné sur Tal joue l'attaque **Cf6-h5** (51 %, contre 12 % pour Petrosian) ; conditionné sur Petrosian, il préfère le regroupement **Cf6-d7** (48 %, contre 9 % pour Tal). En mode Duel, les six joueurs se répartissent en trois coups : Ch5 pour Tal et Fischer, Cfd7 pour Petrosian et Capablanca, d5 pour Kasparov et Carlsen.

Cette position est choisie : c'est celle où Tal et Petrosian divergent le plus parmi 400 positions de milieu de partie tirées au hasard (distance de variation totale 0,44, médiane 0,14). Dans une position typique, les distributions sont bien plus proches (voir [Limites](#limites)).

[Ouvrir cette position dans l'appli](http://127.0.0.1:8000/?vue=analyse&fen=r1bqr1k1/1p3pbp/2pp1np1/p1n5/2PNP3/2N1BPPP/PP4B1/R2QR1K1%20b%20-%20-%201%2013) (serveur lancé en local, voir [Appli de démo](#appli-de-démo)).

## Le modèle

- **Entrée** : 18 plans 8×8 vus du côté du joueur qui a le trait, **sortie** : 4 672 logits (73 types de coups × 64 cases), masqués aux coups légaux (contrat : `docs/CONTRACT.md`).
- **Réseau** : convolution d'entrée, 8 blocs résiduels de 128 filtres (BatchNorm, ReLU), tête de politique 1×1 à 73 canaux. 2,5 M de paramètres.
- **Joueur** : un `nn.Embedding` module chaque bloc par un gain et un biais par canal (FiLM). L'identifiant 0 est le modèle générique (Lichess).
- **Entraînement** : cross-entropy sur les coups légaux, AdamW, mixed precision, learning rate cosine, on garde le meilleur modèle sur le top-1 de validation. Pré-entraînement sur Lichess Elite (identifiant 0), puis fine-tune sur les joueurs iconiques avec leur identifiant.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install stockfish
wandb login
pytest -q
```

Sous Windows : `.venv\Scripts\activate` à la place de `source`, et Stockfish via `winget install Stockfish.Stockfish` (ou le binaire de stockfishchess.org, ajouté au `PATH`).

## Évaluation

```bash
python -m chessstyle.evaluate runs/latest data/processed/icons/test.h5 --out runs/latest/eval_icons.json
```

Affiche le top-1 et le top-3 par joueur et par phase de jeu, puis la matrice croisée : le modèle conditionné sur le joueur i (ligne), évalué sur les vrais coups du joueur j (colonne). Si les styles sont appris, la diagonale est la meilleure de chaque colonne. Les coups 1 à 10 sont mesurés à part. `--wandb PROJET` envoie aussi les chiffres sur W&B.

## Résultats

Données : Lichess Elite 2024-01 (cadence rapide ou plus lente), 2,16 M de positions ; PGN Mentor pour Tal, Petrosian, Capablanca, Fischer, Kasparov et Carlsen, 582 k positions (de 21 k pour Capablanca à 268 k pour Carlsen en entraînement). Split 90/5/5 par partie, jamais par position.

**Pré-entraînement (Lichess)** : top-1 de validation 45,5 % (meilleure époque : la 8e sur 10). Au-delà, le modèle surapprend.

**Fine-tune sur les joueurs iconiques**, jeu de test de 23 344 positions hors ouverture (coups 1 à 10 exclus), mesuré avec `evaluate.py` :

| Modèle | Top-1 | Top-3 |
| --- | --- | --- |
| Pré-entraîné Lichess, identifiant générique | 37,4 % | – |
| Fine-tune, identifiant générique | 38,6 % | – |
| Fine-tune, vrai joueur | **39,3 %** | **67,8 %** |

Le gain total est d'environ **+1,9 point** : +1,3 vient de l'adaptation aux données des maîtres, **+0,6** du conditionnement par joueur. Ces moyennes sont pondérées par le nombre de positions de chaque joueur (calculées à partir de la matrice croisée).

**Style (`style_bench.py`)**, 300 positions par joueur, Stockfish profondeur 10 : le modèle conditionné est plus proche du profil réel du joueur (parts de captures, échecs, coups de pion, dame, roi) que le modèle générique pour **1 joueur sur 6**. La perte moyenne face à Stockfish est de 18 à 28 centipions pour les vrais coups, et de 79 à 118 pour le coup préféré du modèle, conditionné ou non.

## Limites

- **Le style par joueur n'est pas démontré.** Dans la matrice croisée de `evaluate.py`, la diagonale (modèle conditionné sur le bon joueur) n'est la meilleure de sa colonne que dans 1 cas sur 6 en top-1. Les lignes de la matrice sont presque identiques : le modèle ignore en grande partie l'identifiant.
- **Mesures bruitées.** Capablanca n'a que 1 078 positions de test, Fischer 1 439 : un écart de 1 à 2 points est dans le bruit. Pour `style_bench`, avec 300 positions par joueur, chaque part se mesure à ± 3 points près, du même ordre que les différences entre joueurs (contraste moyen de 12 points).
- **Peu de données et un déséquilibre fort** : Carlsen fait 52 % des positions d'entraînement. Équilibrer les joueurs et mélanger 20 % de Lichess n'a pas amélioré le résultat (38,6 % contre 39,3 %).
- **Surapprentissage rapide** au fine-tune : la perte de validation remonte dès la 3e époque, on garde donc le meilleur modèle.
- **Le modèle ne voit que la position**, sans l'historique des coups, et joue nettement moins bien que les humains qu'il imite (perte de 79 à 118 centipions face à 18 à 28).
- **Pas de mesure des sacrifices**, pourtant le style typique de Tal.
- **Pistes** : plus de données par joueur, plus de positions de test, geler le tronc et n'entraîner que les embeddings pour forcer l'usage de l'identifiant.

## Reproduire

Les données et les poids ne sont pas dans Git (`data/` est ignoré). Le fine-tune final a tourné sur Kaggle (GPU).

```bash
# 1. PGN -> Parquet -> HDF5 (split par partie)
python -m chessstyle.ingest lichess data/raw/elite.pgn.zst -o data/interim/lichess.parquet
python -m chessstyle.ingest icons data/raw/Tal.pgn data/raw/Petrosian.pgn data/raw/Capablanca.pgn data/raw/Fischer.pgn data/raw/Kasparov.pgn data/raw/Carlsen.pgn -o data/interim/icons.parquet
python -m chessstyle.build_dataset data/interim/lichess.parquet -o data/processed/lichess
python -m chessstyle.build_dataset data/interim/icons.parquet -o data/processed/icons

# 2. Pré-entraînement Lichess, puis fine-tune iconique
python -m chessstyle.train data/processed/lichess/train.h5 --val data/processed/lichess/val.h5 --workers 2 --batch-size 512 --epochs 10 --weight-decay 0.05 --out runs/lichess_pretrain
python -m chessstyle.train data/processed/icons/train.h5 --val data/processed/icons/val.h5 --init runs/lichess_pretrain --lr 3e-4 --weight-decay 0.05 --epochs 6 --batch-size 512 --out runs/icons_finetune

# 3. Mesures
python -m chessstyle.evaluate runs/icons_finetune data/processed/icons/test.h5 --out runs/icons_finetune/eval.json
python -m chessstyle.style_bench runs/icons_finetune data/processed/icons/test.h5 --out runs/icons_finetune/style.json
```

`train.py` accepte aussi `--balance N` et `--mix-h5` (époques équilibrées par joueur avec une part de Lichess) et `--patience N` (arrêt anticipé).

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
PYTHONPATH=src python -m chessstyle.server --model models/icons_finetune     # http://127.0.0.1:8000
```

`--model` accepte un dossier de run de `train.py` (comme `models/icons_finetune` : poids, `config.yaml`, `players.json`) ou un fichier TorchScript (`python -m chessstyle.model` crée `dummy_model.pt`). Sans `--model`, les poids sont aléatoires.
Pour ouvrir directement une position : `http://127.0.0.1:8000/?vue=analyse&fen=...` (`vue` vaut `joueurs`, `jouer`, `analyse` ou `duel`).
La barre d'évaluation utilise Stockfish s'il est installé (`sudo apt install stockfish`, ou `--stockfish CHEMIN`) ; sinon elle est masquée.
Pour développer l'interface avec rechargement à chaud : lancer le serveur, puis `cd app && npm run dev` (http://localhost:5173, les appels `/api` partent vers le port 8000).

## Règles

- `main` ne reçoit que des PR relues par l'autre, avec la CI verte.
- Une branche par issue : `feat/...`, `fix/...`.
- `encoding.py` ne change qu'avec l'accord des deux et une nouvelle `ENCODING_VERSION`.
- Les décisions vont dans `docs/DECISIONS.md`.
