from __future__ import annotations
import subprocess

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Game import logger
from src.Tile import Tile
from math import sqrt, inf, log
from copy import deepcopy
from random import choice, random


class Node:
    def __init__(self, colour: Colour, move: Move, parent, board: Board, root_colour: Colour):
        # Colour to move in the current turn
        self.colour = colour
        # Last move that was made
        self.move = move
        # Parent of the current node
        self.parent = parent
        # Board position the current node represents, expressed by class Board
        self.board = board
        # Number of vists to the current node via search
        self.visits = 0
        # Number of wins for simulations consisting of this node
        self.wins = 0
        # Children of current node
        self.children = []
        # Untried moves of current node
        self.untried_moves = self.legal_moves(board)
        self.root_colour = root_colour
        # Exploration parameter - can be tuned
        self.c = sqrt(2)

    def legal_moves(self, board: Board):
        moves = []
        for x, col in enumerate(board.tiles):
            for y, tile in enumerate(col):
                if tile.colour is None:
                    moves.append(Move(x, y))
        return moves

    def _neighbours(self, board: Board, x: int, y: int):
        """Return neighbour coordinates using Hex adjacency."""
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        max_x = len(board.tiles)
        max_y = len(board.tiles[0]) if max_x > 0 else 0
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < max_x and 0 <= ny < max_y:
                yield nx, ny

    def _bridge_score(self, board: Board, move: Move, colour: Colour) -> float:
        """
        Local heuristic: prefer moves that
        - touch own stones (extend connections)
        - touch opponent stones (good for blocking)
        - take part in small 'bridge-like' patterns.
        """
        x, y = move.x, move.y
        opp = Colour.opposite(colour)

        own_neighbours = 0
        opp_neighbours = 0
        empty_neighbours = []

        # Count neighbours by type
        for nx, ny in self._neighbours(board, x, y):
            tile = board.tiles[nx][ny]
            if tile.colour == colour:
                own_neighbours += 1
            elif tile.colour == opp:
                opp_neighbours += 1
            else:
                empty_neighbours.append((nx, ny))

        # Base score: extend own groups (3) and block opponent (2)
        score = 3 * own_neighbours + 2 * opp_neighbours

        # Very simple "bridge" approximation:
        # if two empty neighbours share an own-colour neighbour,
        # reward this move as sitting in a locally strong pattern.
        if own_neighbours > 0 and len(empty_neighbours) >= 2:
            own_neighs_sets = []
            for ex, ey in empty_neighbours:
                s = set()
                for nx, ny in self._neighbours(board, ex, ey):
                    tile = board.tiles[nx][ny]
                    if tile.colour == colour:
                        s.add((nx, ny))
                own_neighs_sets.append(s)

            # If any two empties share an own-neighbour, give a small bonus
            for i in range(len(own_neighs_sets)):
                for j in range(i + 1, len(own_neighs_sets)):
                    if own_neighs_sets[i] & own_neighs_sets[j]:
                        score += 1.0

        return score

    def _choose_bridge_move(self, board: Board, moves, colour: Colour) -> Move:
        """Pick a move biased by the local bridge heuristic."""
        best_score = -inf
        best_moves = []

        for m in moves:
            s = self._bridge_score(board, m, colour)
            if s > best_score:
                best_score = s
                best_moves = [m]
            elif s == best_score:
                best_moves.append(m)

        if not best_moves:
            return choice(moves)
        return choice(best_moves)

    def ucb(self, child: Node):
        if child.visits == 0:
            return inf
        return (child.wins/child.visits) + self.c * sqrt(log(self.visits)/child.visits)

    def best_child(self):
        return max(self.children, key=lambda x: self.ucb(x))

    def select(self):
        # TODO: What if there are multiple children of the same UCB? The selection might not be truly random. It might also be worth it to implement a heuristic for this

        # If there are no untried moves, iterate until a child with untried moves is reached
        node = self
        while not node.untried_moves and node.children:
            node = node.best_child()
        return node

    def expand(self):
        # Make a random move from selected node
        move = self.untried_moves.pop()
        # TODO: deepcopy is very slow
        new_board = deepcopy(self.board)
        # Make move on board
        new_board.set_tile_colour(move.x, move.y, self.colour)
        new_node = Node(Colour.opposite(self.colour), move,
                        self, new_board, self.root_colour)
        self.children.append(new_node)
        return new_node

    def simulate(self):
        # While the game hasn't ended, keep on playing random moves
        # TODO: Add a heuristic to select non-random moves

        # TODO: again deepcopy here is very slow
        new_board = deepcopy(self.board)
        colour = self.colour

        while True:
            if new_board.has_ended(Colour.RED):
                return Colour.RED
            if new_board.has_ended(Colour.BLUE):
                return Colour.BLUE

            moves = self.legal_moves(new_board)
            if not moves:
                # No legal moves left: treat as loss for current player
                return Colour.opposite(colour)

            # mostly bridge-biased playouts, sometimes pure random for exploration
            if random() < 0.8:
                move = self._choose_bridge_move(new_board, moves, colour)
            else:
                move = choice(moves)

            new_board.set_tile_colour(move.x, move.y, colour)
            colour = Colour.opposite(colour)

    def backpropagate(self, winner: Colour):
        self.visits += 1
        if winner == self.root_colour:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(winner)

    # TODO: make search time based to fit with time constraints of CW
    def search(self, iterations):
        for i in range(iterations):
            print("Iteration", i)
            node = self.select()
            if node.untried_moves:
                node = node.expand()
            winner = node.simulate()
            node.backpropagate(winner)

        best_child = max(self.children, key=lambda x: x.visits)
        return best_child.move


class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        # Colour: red or blue - red moves first.
        super().__init__(colour)

        # self.agent_process = subprocess.Popen(
        #     ["./agents/MCTSAgent/mcts-hex"],
        #     stdout=subprocess.PIPE,
        #     stdin=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     text=True,
        #     bufsize=1
        # )

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """

        # rows = board.tiles
        # board_strings = []
        # for row in rows:
        #     row_string = ""
        #     for tile in row:
        #         colour = tile.colour
        #         if colour is None:
        #             t = "0"
        #         else:
        #             t = colour.get_char()
        #         row_string += t
        #     board_strings.append(row_string)
        # board_string = ",".join(board_strings)
        #
        # if opp_move is None:
        #     command = f"START;;{board_string};{turn};"
        # elif opp_move.x == -1 and opp_move.y == -1:
        #     command = f"SWAP;;{board_string};{turn};"
        # else:
        #     command = f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"
        #
        # self.agent_process.stdin.write(command + "\n")
        # self.agent_process.stdin.flush()

        root = Node(self.colour, opp_move, None, board, self.colour)
        iterations = 20
        response = root.search(iterations)
        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        return response
