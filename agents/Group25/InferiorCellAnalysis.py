# agents/Group25/InferiorCellAnalysis.py
from __future__ import annotations
from typing import List, Tuple

from src.Colour import Colour
from agents.Group25.Bitboard import MASKS

# Public API: filter_moves(bitboard, player_colour, move_list) -> List[(x,y)]
#             is_inferior_move(bitboard, player_colour, move) -> bool

def _opponent_bits(bitboard, player_colour: Colour) -> int:
    return bitboard.red if player_colour == Colour.BLUE else bitboard.blue

def _occupied_bits(bitboard) -> int:
    return bitboard.red | bitboard.blue

def _popcount(x: int) -> int:
    # small fast popcount for Python ints
    return x.bit_count()

def is_dead_cell(bitboard, player_colour: Colour, idx: int) -> bool:
    """
    Conservative dead-cell test.

    A cell is considered dead iff:
      1) It is empty.
      2) It does NOT touch either of P's goal sides (orientation-aware).
      3) Every neighbour is occupied (no empty neighbour).
      4) Every occupied neighbour belongs to the opponent, AND there are at least TWO opponent neighbours.

    This reduces false positives compared to an aggressive 'all-neighbours-opponent' check
    by requiring at least two opponent neighbours. It also disables pruning very early
    in the game (see filter_moves), which prevents opening mistakes.
    """
    size = bitboard.size
    x = idx // size
    y = idx % size

    # orientation check: if cell touches player's goal sides, it can't be dead
    if player_colour == Colour.RED:
        # RED connects TOP (x=0) -> BOTTOM (x=size-1)
        if x == 0 or x == size - 1:
            return False
    else:
        # BLUE connects LEFT (y=0) -> RIGHT (y=size-1)
        if y == 0 or y == size - 1:
            return False

    bit = 1 << idx
    occ = _occupied_bits(bitboard)
    # must be empty
    if (occ & bit) != 0:
        # occupied squares are not legal moves — treat as not-dead for pruning safety
        return False

    neigh_mask = MASKS[idx]

    # if any neighbour is empty -> not dead
    if (neigh_mask & ~occ) != 0:
        return False

    # all neighbours occupied; check whether they are all opponent AND at least 2 opponent neighbours
    opp_bits = _opponent_bits(bitboard, player_colour)
    neigh_occ = neigh_mask & occ
    neigh_opp = neigh_mask & opp_bits

    # require all occupied neighbours to be opponent AND at least 2 opponent neighbours
    if neigh_occ != 0 and neigh_occ == neigh_opp and _popcount(neigh_opp) >= 2:
        return True

    return False


def is_inferior_move(bitboard, player_colour: Colour, move: Tuple[int, int]) -> bool:
    """
    Conservative single-move inferior test.

    Currently only uses dead-cell analysis. Returns True if the move is considered inferior.
    """
    x, y = move
    idx = bitboard.index(x, y)

    # defensive: if occupied, mark as inferior (shouldn't happen in normal codepaths)
    occ = _occupied_bits(bitboard)
    if ((occ >> idx) & 1) != 0:
        return True

    if is_dead_cell(bitboard, player_colour, idx):
        return True

    # no other heuristics enabled (keeps ICA conservative)
    return False


def filter_moves(bitboard, player_colour: Colour, move_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Return subset of move_list that survive ICA pruning.

    Safety measures included:
      - When the board is very empty (<= 6 stones total) do NOT prune at all.
      - If pruning would remove all moves, return the original list (avoid starving the search).
    """
    # Very early in the game: do not prune (avoids opening blunders)
    occ = _occupied_bits(bitboard)
    total_stones = _popcount(occ)
    if total_stones <= 6:
        return list(move_list)

    survivors: List[Tuple[int, int]] = []
    for m in move_list:
        if not is_inferior_move(bitboard, player_colour, m):
            survivors.append(m)

    # If ICA filtered everything, return original list
    if not survivors:
        return list(move_list)

    return survivors