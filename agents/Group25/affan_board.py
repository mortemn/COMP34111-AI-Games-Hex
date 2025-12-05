# agents/Group25/affan_board.py

from __future__ import annotations
from typing import List, Tuple

from src.Colour import Colour
from src.Board import Board


class AffanBoardState:
    """
    Lightweight Hex board for MCTS.

    - Uses integers: 0 = empty, 1 = RED, 2 = BLUE.
    - Maintains a list of empty cells for fast random move choice.
    - Uses Union-Find (DSU) with virtual edges for O(α(N)) winner checks.
    """

    EMPTY = 0
    RED = 1
    BLUE = 2

    # Hex neighbours
    NEIGHBOURS: list[tuple[int, int]] = [
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
    ]

    def __init__(self, size: int = 11):
        self.size = size
        self.grid: list[list[int]] = [[self.EMPTY] * size for _ in range(size)]

        # Maintain list of empty cells (updated on play)
        self.empty_cells: list[tuple[int, int]] = [
            (x, y) for x in range(size) for y in range(size)
        ]
        # NEW: map each (x, y) to its index in empty_cells for O(1) removal
        self.empty_index: dict[tuple[int, int], int] = {
            (x, y): i for i, (x, y) in enumerate(self.empty_cells)
        }

        # DSU parents: +2 for virtual nodes
        n = size * size
        self.parent_red: list[int] = list(range(n + 2))
        self.parent_blue: list[int] = list(range(n + 2))

        self.TOP = n
        self.BOTTOM = n + 1
        self.LEFT = n
        self.RIGHT = n + 1

        # Winner as Colour or None
        self._winner: Colour | None = None

        # Precompute neighbours once per board
        self.neighbours: list[list[list[tuple[int, int]]]] = [
            [[] for _ in range(size)] for _ in range(size)
        ]
        for x in range(size):
            for y in range(size):
                nbrs: list[tuple[int, int]] = []
                for dx, dy in self.NEIGHBOURS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        nbrs.append((nx, ny))
                self.neighbours[x][y] = nbrs

    # ---------- internal DSU helpers ----------

    def _index(self, x: int, y: int) -> int:
        return x * self.size + y

    def _find(self, parent: list[int], a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(self, parent: list[int], a: int, b: int) -> None:
        ra = self._find(parent, a)
        rb = self._find(parent, b)
        if ra != rb:
            parent[rb] = ra

    # ---------- cloning ----------

    def copy(self) -> "AffanBoardState":
        """Fast clone for MCTS playouts, avoiding __init__ overhead."""
        new = object.__new__(AffanBoardState)

        # Scalars
        new.size = self.size
        new.TOP = self.TOP
        new.BOTTOM = self.BOTTOM
        new.LEFT = self.LEFT
        new.RIGHT = self.RIGHT

        # Arrays / state
        new.grid = [row[:] for row in self.grid]
        new.empty_cells = self.empty_cells[:]      # shallow copy OK
        new.empty_index = self.empty_index.copy()
        new.parent_red = self.parent_red[:]
        new.parent_blue = self.parent_blue[:]
        new._winner = self._winner

        # Neighbours are static: safe to share reference
        new.neighbours = self.neighbours

        return new

    # ---------- core operations ----------

    def play(self, x: int, y: int, colour: Colour) -> None:
        """
        Apply a move for `colour` (src.Colour.RED / BLUE).
        Updates grid, DSU, empty_cells and winner.
        """
        if self.grid[x][y] != self.EMPTY:
            raise ValueError("Invalid move: cell already occupied")

        if colour == Colour.RED:
            val = self.RED
        elif colour == Colour.BLUE:
            val = self.BLUE
        else:
            raise ValueError(f"Invalid colour for play: {colour}")

        self.grid[x][y] = val

        # --- O(1) removal from empty_cells using swap-with-last ---
        coord = (x, y)
        idx = self.empty_index.pop(coord, None)
        if idx is not None:
            last_coord = self.empty_cells[-1]
            # Move last element into the removed slot (if not already last)
            self.empty_cells[idx] = last_coord
            # Update its index in the map
            self.empty_index[last_coord] = idx
            # Pop the last element
            self.empty_cells.pop()
        # If idx is None, it means it wasn't in empty_cells (shouldn't happen
        # in normal play), but we just skip instead of crashing.

        idx_flat = self._index(x, y)

        if val == self.RED:
            parent = self.parent_red
            # connect to virtual borders
            if x == 0:
                self._union(parent, idx_flat, self.TOP)
            if x == self.size - 1:
                self._union(parent, idx_flat, self.BOTTOM)

            # connect to neighbouring reds
            for dx, dy in self.NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx][ny] == self.RED:
                        self._union(parent, idx_flat, self._index(nx, ny))

            # check win
            if self._find(parent, self.TOP) == self._find(parent, self.BOTTOM):
                self._winner = Colour.RED

        else:  # BLUE
            parent = self.parent_blue

            if y == 0:
                self._union(parent, idx_flat, self.LEFT)
            if y == self.size - 1:
                self._union(parent, idx_flat, self.RIGHT)

            for dx, dy in self.NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx][ny] == self.BLUE:
                        self._union(parent, idx_flat, self._index(nx, ny))

            if self._find(parent, self.LEFT) == self._find(parent, self.RIGHT):
                self._winner = Colour.BLUE

    # ---------- queries ----------

    def get_empty_cells(self) -> list[tuple[int, int]]:
        return self.empty_cells

    def is_terminal(self) -> bool:
        return self._winner is not None

    def get_winner(self) -> Colour | None:
        return self._winner

    # ---------- construction from engine Board ----------

    @classmethod
    def from_engine_board(cls, board: Board) -> "AffanBoardState":
        """
        Build a fast AffanBoardState from the engine's Board,
        including DSU connectivity and empty_cells.
        """
        size = board.size
        bs = cls(size)

        # reset everything
        bs.grid = [[cls.EMPTY] * size for _ in range(size)]
        bs.empty_cells = []
        bs.empty_index = {}
        n = size * size
        bs.parent_red = list(range(n + 2))
        bs.parent_blue = list(range(n + 2))
        bs._winner = None

        for x in range(size):
            for y in range(size):
                tile_colour = board.tiles[x][y].colour
                idx_flat = bs._index(x, y)

                if tile_colour is None:
                    bs.grid[x][y] = cls.EMPTY
                    coord = (x, y)
                    bs.empty_index[coord] = len(bs.empty_cells)
                    bs.empty_cells.append(coord)
                    continue

                if tile_colour == Colour.RED:
                    bs.grid[x][y] = cls.RED
                    # connect to virtual borders
                    if x == 0:
                        bs._union(bs.parent_red, idx_flat, bs.TOP)
                    if x == size - 1:
                        bs._union(bs.parent_red, idx_flat, bs.BOTTOM)
                else:  # BLUE
                    bs.grid[x][y] = cls.BLUE
                    if y == 0:
                        bs._union(bs.parent_blue, idx_flat, bs.LEFT)
                    if y == size - 1:
                        bs._union(bs.parent_blue, idx_flat, bs.RIGHT)

        # now connect neighbours of same colour
        for x in range(size):
            for y in range(size):
                val = bs.grid[x][y]
                if val == cls.EMPTY:
                    continue

                idx_flat = bs._index(x, y)
                if val == cls.RED:
                    for dx, dy in cls.NEIGHBOURS:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if bs.grid[nx][ny] == cls.RED:
                                bs._union(bs.parent_red, idx_flat,
                                          bs._index(nx, ny))
                else:  # BLUE
                    for dx, dy in cls.NEIGHBOURS:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if bs.grid[nx][ny] == cls.BLUE:
                                bs._union(bs.parent_blue, idx_flat,
                                          bs._index(nx, ny))

        # compute winner if any
        if bs._find(bs.parent_red, bs.TOP) == bs._find(bs.parent_red, bs.BOTTOM):
            bs._winner = Colour.RED
        elif bs._find(bs.parent_blue, bs.LEFT) == bs._find(bs.parent_blue, bs.RIGHT):
            bs._winner = Colour.BLUE

        return bs
