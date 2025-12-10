# agents/Group25/InferiorCellAnalysis.py
from __future__ import annotations
from typing import List, Tuple

from src.Colour import Colour
from agents.Group25.Bitboard import MASKS

# Public API:
#   filter_moves(bitboard, player_colour, move_list) -> list[(x,y)]
#   is_inferior_move(bitboard, player_colour, move) -> bool
#   is_dead_cell(bitboard, player_colour, idx) -> bool
#   is_captured_cell(bitboard, player_colour, idx) -> bool

def _opponent(player_colour: Colour) -> Colour:
    return Colour.RED if player_colour == Colour.BLUE else Colour.BLUE


def is_dead_cell(bitboard, player_colour: Colour, idx: int) -> bool:
    """
    Conservative dead-cell test (bitboard-based).

    Conservative definition used here:
      - If the cell touches either of player's goal sides, it is NOT dead.
      - Otherwise, the cell is dead iff every neighbour is occupied AND
        every occupied neighbour belongs to the opponent (i.e. no empty neighbours
        and no friendly neighbours).

    Returns True if the cell is considered dead (safe-to-prune).
    """
    size = bitboard.size
    x = idx // size
    y = idx % size

    # If the cell lies on either of the player's goal sides, it cannot be dead.
    if player_colour == Colour.RED:
        # RED connects TOP (x=0) -> BOTTOM (x=size-1)
        if x == 0 or x == size - 1:
            return False
    else:  # BLUE
        # BLUE connects LEFT (y=0) -> RIGHT (y=size-1)
        if y == 0 or y == size - 1:
            return False

    # Defensive: if the cell is already occupied, treat as inferior (shouldn't happen in legal_moves)
    bit = 1 << idx
    occ = bitboard.red | bitboard.blue
    if occ & bit:
        return True

    neigh_mask = MASKS[idx]

    # If any neighbour is empty -> cell is not dead
    if (neigh_mask & ~occ) != 0:
        return False

    # All neighbours are occupied; check whether all occupied neighbour bits belong to opponent
    opp_bits = bitboard.red if player_colour == Colour.BLUE else bitboard.blue

    if (neigh_mask & occ) == (neigh_mask & opp_bits):
        # every occupied neighbour belongs to opponent -> cell is dead
        return True

    return False


def is_captured_cell(bitboard, player_colour: Colour, idx: int) -> bool:
    """
    Conservative 'captured' test:
      - A cell is 'captured' for player P if there are NO opponent neighbours.
    (This is a conservative proxy: if there are zero opponent neighbours, the cell
     is likely to be captured or at least safe to consider for pruning in some solvers.)
    """
    occ = bitboard.red | bitboard.blue
    bit = 1 << idx
    if occ & bit:
        # occupied cells are not legal; treat as inferior by default
        return True

    neigh_mask = MASKS[idx]
    opp_bits = bitboard.red if player_colour == Colour.BLUE else bitboard.blue

    # If any neighbour is an opponent bit, it's not captured
    if (neigh_mask & opp_bits) != 0:
        return False

    # No opponent neighbours -> treat as captured (conservative)
    return True


# Future more aggressive checks (stubs). Disabled by default.
def _is_vulnerable_cell(bitboard, player_colour: Colour, idx: int) -> bool:
    """
    Placeholder for a vulnerability detection (e.g. will become dead after an opponent reply).
    Not enabled in filter_moves by default because it's less conservative and risks pruning
    useful moves. Implement carefully before enabling.
    """
    return False


def _is_capture_dominated(bitboard, player_colour: Colour, idx: int) -> bool:
    """
    Placeholder for capture-dominated test.
    Disabled by default.
    """
    return False


def is_inferior_move(bitboard, player_colour: Colour, move: Tuple[int, int]) -> bool:
    """
    Conservative test whether single move (x,y) is inferior.
    Current checks:
      - dead cell (primary)
      - captured cell (secondary, optional)
    """
    x, y = move
    idx = bitboard.index(x, y)

    # Defensive: if the cell is already occupied, it's inferior (caller should avoid)
    occ = bitboard.red | bitboard.blue
    if (occ >> idx) & 1:
        return True

    # Dead cells are safe to prune
    if is_dead_cell(bitboard, player_colour, idx):
        return True

    # Captured cells are optional prune; keep conservative: prune if clearly captured
    if is_captured_cell(bitboard, player_colour, idx):
        return True

    # Do NOT prune on vulnerable/capture-dominated here (too risky)
    # if _is_vulnerable_cell(bitboard, player_colour, idx):
    #     return True

    return False


def filter_moves(bitboard, player_colour: Colour, move_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for m in move_list:
        if not is_inferior_move(bitboard, player_colour, m):
            out.append(m)
    return out
