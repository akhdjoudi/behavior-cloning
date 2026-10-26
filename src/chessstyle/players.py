"""Liste figée des joueurs imités. On ajoute à la fin, on ne renumérote jamais.

La liste est sauvegardée avec chaque checkpoint (players.json).
Liste v1 proposée, à valider après le comptage des parties disponibles (au moins 1 000 par joueur).
"""

GENERIC = 0

PLAYERS = [
    "generique",  # 0 : positions Lichess du pré-entraînement
    "Tal",
    "Petrosian",
    "Capablanca",
    "Fischer",
    "Kasparov",
    "Carlsen",
]

PLAYER_ID = {name: i for i, name in enumerate(PLAYERS)}
