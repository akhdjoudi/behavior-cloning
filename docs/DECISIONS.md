# Journal des décisions

Une ligne par décision : date, choix, raison.

| Date | Décision | Raison |
| --- | --- | --- |
| 2026-10-26 | Échecs seulement pour la v1, interface `Game` générique | Garder le périmètre ; le go pourra s'ajouter sans réécriture |
| 2026-10-26 | Encodage AlphaZero 4 672 classes plutôt que 4 096 | 4 096 ne couvre pas les sous-promotions |
| 2026-10-26 | Un seul modèle conditionné par un embedding de joueur | Plusieurs joueurs iconiques au choix ; les joueurs partagent leurs données |
| 2026-10-27 | Liste des joueurs v1 gardée telle quelle, avec un warning sous 1 000 parties | Capablanca (597) et Fischer (785) sont sous le seuil mais PGN Mentor n'en a pas plus |
| 2026-10-27 | Statistiques d'ingestion (PGN Mentor) : Tal 2 371 parties / 84 262 positions, Petrosian 1 895 / 66 412, Capablanca 597 / 23 346, Fischer 786 / 32 159, Kasparov 1 963 / 74 781, Carlsen 6 488 / 300 640 | Positions = coups joués par le joueur imité ; parties blitz/bullet exclues |
