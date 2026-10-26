"""Contrat d'interface v1 : encodage des positions et des coups d'échecs.

Toute modification exige l'accord des deux rôles et l'incrémentation de ENCODING_VERSION.
Spécification lisible : docs/CONTRACT.md.
"""

import chess
import numpy as np

ENCODING_VERSION = 1

NUM_PLANES = 18
NUM_MOVE_TYPES = 73
NUM_ACTIONS = NUM_MOVE_TYPES * 64  # 4672

# Plans d'entrée
PLANE_CASTLING = 12  # 12 à 15 : mon petit roque, mon grand roque, son petit roque, son grand roque
PLANE_EN_PASSANT = 16
PLANE_HALFMOVE = 17

# Types de coup : 0-55 coups « de dame » (direction * 7 + distance - 1),
# 56-63 coups de cavalier, 64-72 sous-promotions (pièce * 3 + direction).
QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]  # (dfile, drank) : N, NE, E, SE, S, SW, W, NW
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
FIRST_KNIGHT_TYPE = 56
FIRST_UNDERPROMO_TYPE = 64


def _canonical(board: chess.Board) -> chess.Board:
    """Plateau vu du joueur qui a le trait : on retourne et on inverse les couleurs quand c'est aux Noirs."""
    return board if board.turn == chess.WHITE else board.mirror()


def _canonical_move(move: chess.Move, board: chess.Board) -> chess.Move:
    if board.turn == chess.WHITE:
        return move
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)


class ChessGame:
    board_shape = (NUM_PLANES, 8, 8)
    num_actions = NUM_ACTIONS

    def encode(self, board: chess.Board) -> np.ndarray:
        b = _canonical(board)
        planes = np.zeros(self.board_shape, dtype=np.uint8)

        for piece_type in chess.PIECE_TYPES:
            for color, offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
                for sq in b.pieces(piece_type, color):
                    planes[offset + piece_type - 1, chess.square_rank(sq), chess.square_file(sq)] = 1

        rights = [
            b.has_kingside_castling_rights(chess.WHITE),
            b.has_queenside_castling_rights(chess.WHITE),
            b.has_kingside_castling_rights(chess.BLACK),
            b.has_queenside_castling_rights(chess.BLACK),
        ]
        for i, has_right in enumerate(rights):
            if has_right:
                planes[PLANE_CASTLING + i] = 1

        # Seulement si la prise en passant est réellement jouable, pour qu'une même position ait un seul encodage
        if b.ep_square is not None and b.has_legal_en_passant():
            planes[PLANE_EN_PASSANT, chess.square_rank(b.ep_square), chess.square_file(b.ep_square)] = 1

        planes[PLANE_HALFMOVE] = min(b.halfmove_clock, 100)
        return planes

    def move_to_index(self, move: chess.Move, board: chess.Board) -> int:
        m = _canonical_move(move, board)
        dfile = chess.square_file(m.to_square) - chess.square_file(m.from_square)
        drank = chess.square_rank(m.to_square) - chess.square_rank(m.from_square)

        if m.promotion is not None and m.promotion != chess.QUEEN:
            move_type = FIRST_UNDERPROMO_TYPE + UNDERPROMOTIONS.index(m.promotion) * 3 + (dfile + 1)
        elif (dfile, drank) in KNIGHT_DELTAS:
            move_type = FIRST_KNIGHT_TYPE + KNIGHT_DELTAS.index((dfile, drank))
        else:
            distance = max(abs(dfile), abs(drank))
            direction = QUEEN_DIRECTIONS.index((dfile // distance, drank // distance))
            move_type = direction * 7 + distance - 1

        return move_type * 64 + m.from_square

    def index_to_move(self, idx: int, board: chess.Board) -> chess.Move:
        if not 0 <= idx < NUM_ACTIONS:
            raise ValueError(f"index hors limites : {idx}")
        move_type, from_sq = divmod(idx, 64)
        b = _canonical(board)
        promotion = None

        if move_type >= FIRST_UNDERPROMO_TYPE:
            piece, direction = divmod(move_type - FIRST_UNDERPROMO_TYPE, 3)
            dfile, drank = direction - 1, 1
            promotion = UNDERPROMOTIONS[piece]
        elif move_type >= FIRST_KNIGHT_TYPE:
            dfile, drank = KNIGHT_DELTAS[move_type - FIRST_KNIGHT_TYPE]
        else:
            direction, distance = divmod(move_type, 7)
            step_file, step_rank = QUEEN_DIRECTIONS[direction]
            dfile, drank = step_file * (distance + 1), step_rank * (distance + 1)

        to_file = chess.square_file(from_sq) + dfile
        to_rank = chess.square_rank(from_sq) + drank
        if not (0 <= to_file < 8 and 0 <= to_rank < 8):
            raise ValueError(f"l'index {idx} sort de l'échiquier")
        to_sq = chess.square(to_file, to_rank)

        # Un pion qui atteint la dernière rangée par un coup « de dame » est promu en dame
        if promotion is None and b.piece_type_at(from_sq) == chess.PAWN and to_rank == 7:
            promotion = chess.QUEEN

        if board.turn == chess.BLACK:
            from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
        return chess.Move(from_sq, to_sq, promotion)

    def legal_mask(self, board: chess.Board) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for move in board.legal_moves:
            mask[self.move_to_index(move, board)] = True
        return mask
