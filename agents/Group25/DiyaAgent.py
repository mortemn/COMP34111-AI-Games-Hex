# agents/Group25/MCTSAgent.py
from __future__ import annotations
import time
import math
import random
from typing import List, Tuple

from src.AgentBase import AgentBase
from src.Move import Move
from src.Colour import Colour


# =========================================================
#                 LIGHTWEIGHT BOARDSTATE
# =========================================================
class BoardState:
    EMPTY = 0
    RED = 1
    BLUE = 2

    NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    def __init__(self, size: int = 11):
        self.size = size
        self.grid = [[BoardState.EMPTY] * size for _ in range(size)]
        self.empty_cells = [(x, y) for x in range(size) for y in range(size)]

        n = size * size
        self.parent_red = list(range(n + 2))
        self.parent_blue = list(range(n + 2))

        self.TOP = n
        self.BOTTOM = n + 1
        self.LEFT = n
        self.RIGHT = n + 1

        self._winner = None

    def _index(self, x: int, y: int) -> int:
        return x * self.size + y

    def _find(self, parent, a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(self, parent, a, b):
        ra = self._find(parent, a)
        rb = self._find(parent, b)
        if ra != rb:
            parent[rb] = ra

    def copy(self):
        new = BoardState(self.size)
        new.grid = [row[:] for row in self.grid]
        new.empty_cells = self.empty_cells[:]
        new.parent_red = self.parent_red[:]
        new.parent_blue = self.parent_blue[:]
        new._winner = self._winner
        return new
    
    def play(self, x, y, colour):
        if self.grid[x][y] != BoardState.EMPTY:
            raise ValueError("Invalid move - cell already occupied")

        self.grid[x][y] = colour
        try:
            self.empty_cells.remove((x, y))
        except ValueError:
            pass

        idx = self._index(x, y)

        if colour == BoardState.RED:
            parent = self.parent_red
            if x == 0:
                self._union(parent, idx, self.TOP)
            if x == self.size - 1:
                self._union(parent, idx, self.BOTTOM)
            for dx, dy in BoardState.NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx][ny] == BoardState.RED:
                        self._union(parent, idx, self._index(nx, ny))
            if self._find(parent, self.TOP) == self._find(parent, self.BOTTOM):
                self._winner = Colour.RED

        else:  # BLUE
            parent = self.parent_blue
            if y == 0:
                self._union(parent, idx, self.LEFT)
            if y == self.size - 1:
                self._union(parent, idx, self.RIGHT)
            for dx, dy in BoardState.NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx][ny] == BoardState.BLUE:
                        self._union(parent, idx, self._index(nx, ny))
            if self._find(parent, self.LEFT) == self._find(parent, self.RIGHT):
                self._winner = Colour.BLUE

    def get_empty_cells(self):
        return self.empty_cells

    def is_terminal(self):
        return self._winner is not None

    def check_winner_fast(self):
        return self._winner

    @staticmethod
    def from_engine_board(board):
        size = board.size
        bs = BoardState(size)

        bs.empty_cells = []

        for x in range(size):
            for y in range(size):
                tile = board.tiles[x][y].colour
                if tile is None:
                    bs.grid[x][y] = BoardState.EMPTY
                    bs.empty_cells.append((x, y))
                elif tile == Colour.RED:
                    bs.grid[x][y] = BoardState.RED
                else:
                    bs.grid[x][y] = BoardState.BLUE

        # rebuild UF structure
        n = size * size
        bs.parent_red = list(range(n + 2))
        bs.parent_blue = list(range(n + 2))

        for x in range(size):
            for y in range(size):
                val = bs.grid[x][y]
                idx = bs._index(x, y)
                if val == BoardState.RED:
                    if x == 0: bs._union(bs.parent_red, idx, bs.TOP)
                    if x == size - 1: bs._union(bs.parent_red, idx, bs.BOTTOM)
                    for dx, dy in BoardState.NEIGHBOURS:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if bs.grid[nx][ny] == BoardState.RED:
                                bs._union(bs.parent_red, idx, bs._index(nx, ny))
                elif val == BoardState.BLUE:
                    if y == 0: bs._union(bs.parent_blue, idx, bs.LEFT)
                    if y == size - 1: bs._union(bs.parent_blue, idx, bs.RIGHT)
                    for dx, dy in BoardState.NEIGHBOURS:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if bs.grid[nx][ny] == BoardState.BLUE:
                                bs._union(bs.parent_blue, idx, bs._index(nx, ny))

        if bs._find(bs.parent_red, bs.TOP) == bs._find(bs.parent_red, bs.BOTTOM):
            bs._winner = Colour.RED
        elif bs._find(bs.parent_blue, bs.LEFT) == bs._find(bs.parent_blue, bs.RIGHT):
            bs._winner = Colour.BLUE

        return bs


# =========================================================
#                      MCTS NODE
# =========================================================
class Node:
    def __init__(self, state: BoardState, player_to_move: Colour,
                 parent=None, move=None, root_player=None):

        self.state = state
        self.parent = parent
        self.move = move  
        self.player_to_move = player_to_move
        self.root_player = root_player if root_player else player_to_move

        self.children = []
        self.untried_moves = state.get_empty_cells()[:]  # list of (x,y)

        self.visits = 0
        self.wins = 0
        self.c = math.sqrt(2)

    def uct(self, child):
        if child.visits == 0:
            return float("inf")
        return (child.wins / child.visits) + self.c * math.sqrt(math.log(self.visits) / child.visits)

    def best_child(self):
        return max(self.children, key=lambda c: self.uct(c))

    def select(self):
        node = self
        while not node.untried_moves and node.children:
            node = node.best_child()
        return node

    def expand(self):
        mx, my = self.untried_moves.pop()
        new_state = self.state.copy()

        colour_int = BoardState.RED if self.player_to_move == Colour.RED else BoardState.BLUE
        new_state.play(mx, my, colour_int)

        child = Node(
            new_state,
            Colour.opposite(self.player_to_move),
            parent=self,
            move=Move(mx, my),
            root_player=self.root_player,
        )
        self.children.append(child)
        return child

    def simulate(self):
        state = self.state.copy()
        cur = self.player_to_move

        while True:
            w = state.check_winner_fast()
            if w is not None:
                return w

            moves = state.get_empty_cells()
            if not moves:
                return Colour.opposite(cur)

            mx, my = random.choice(moves)
            colour_int = BoardState.RED if cur == Colour.RED else BoardState.BLUE
            state.play(mx, my, colour_int)
            cur = Colour.opposite(cur)

    def backpropagate(self, winner):
        self.visits += 1
        if winner == self.root_player:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(winner)

    def run_iterations(self, max_iterations, max_time=None, debug=False):
        start = time.time()
        iterations = 0

        for _ in range(max_iterations):
            iterations += 1

            if max_time is not None:
                if time.time() - start >= max_time:
                    break

            node = self.select()
            if node.untried_moves:
                node = node.expand()
            winner = node.simulate()
            node.backpropagate(winner)

        if debug:
            elapsed = time.time() - start
            print(f"[MCTS] Iterations: {iterations}, Time: {elapsed:.4f}s")

        if not self.children:
            # fallback
            mx, my = random.choice(self.state.get_empty_cells())
            return Move(mx, my)

        best = max(self.children, key=lambda c: c.visits)
        return best.move


# =========================================================
#                     MCTS AGENT
# =========================================================
class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

        self.max_iterations = 15000     # run EXACTLY this many iterations unless time exceeded
        self.max_time = 3.0            # optional time cap (None = no cap)
        self.debug = True               # print iteration + time

    def make_move(self, turn: int, board, opp_move):
        root_state = BoardState.from_engine_board(board)

        root = Node(
            state=root_state,
            player_to_move=self.colour,
            parent=None,
            move=None,
            root_player=self.colour
        )

        best_move = root.run_iterations(
            max_iterations=self.max_iterations,
            max_time=self.max_time,
            debug=self.debug
        )

        return best_move
