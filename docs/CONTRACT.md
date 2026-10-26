# Contrat d'interface v1

Implémentation : `src/chessstyle/encoding.py` (`ChessGame`). Tests : `tests/test_encoding.py`.
Toute modification exige l'accord des deux rôles, l'incrémentation de `ENCODING_VERSION` et des tests à jour.

## Entrée : tenseur `(18, 8, 8)` en `uint8`

Toujours vu du côté du joueur qui a le trait : quand c'est aux Noirs, on encode `board.mirror()`.
Indexation `planes[plan, rangée, colonne]`, rangée 0 = première rangée du joueur qui a le trait.

| Plans | Contenu |
| --- | --- |
| 0 à 5 | Mes pièces : pion, cavalier, fou, tour, dame, roi |
| 6 à 11 | Pièces adverses, même ordre |
| 12 à 15 | Droits de roque : mon petit, mon grand, son petit, son grand |
| 16 | Case de prise en passant, seulement si la prise est légale |
| 17 | `min(halfmove_clock, 100)` sur tout le plan |

## Sortie : 4 672 classes (73 types × 64 cases de départ)

`index = type * 64 + case_de_départ` (cases python-chess : a1 = 0, h8 = 63, dans le repère du joueur qui a le trait).

| Types | Coups |
| --- | --- |
| 0 à 55 | Coups « de dame » : `direction * 7 + distance - 1`, directions N, NE, E, SE, S, SW, W, NW |
| 56 à 63 | Coups de cavalier, deltas (colonne, rangée) : (1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2) |
| 64 à 72 | Sous-promotions : `pièce * 3 + (dcolonne + 1)`, pièces cavalier, fou, tour |

Les promotions en dame sont des coups « de dame » normaux. Le roque est le coup du roi de deux cases.

## Identifiant de joueur

Seconde entrée du modèle, définie dans `src/chessstyle/players.py`. 0 = générique (Lichess).

## Exemples de référence

| Position | Coup | Index |
| --- | --- | --- |
| Position initiale, trait aux Blancs | `e2e4` | 76 (type 1 = N distance 2, case e2 = 12) |
| Après `1.e4`, trait aux Noirs | `e7e5` | 76 (après mirroring, e7 devient e2) |
