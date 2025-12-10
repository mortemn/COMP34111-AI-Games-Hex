from __future__ import annotations
import subprocess
import time
from agents.Group25.Bitboard import Bitboard, convert_bitboard

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Game import logger
from src.Tile import Tile
from math import sqrt, inf, log
from copy import deepcopy
from random import choice

TIME_LIMIT = 3 * 60

def time_allocator(turn: int, board: Bitboard, time_remaining: float):
    if time_remaining <= 0.01:
        return 0.01

    board_area = board.size * board.size
    moves_played = turn - 1

    our_moves_left = (board_area - moves_played) // 2

    baseline = time_remaining / our_moves_left

    if turn <= 5:
        # Opening
        phase = 3
    elif turn <= 10:
        # Early middlegame
        phase = 2
    elif turn <= 30:
        phase = 1.5
    elif turn <= 40:
        phase = 1.0
    else:
        phase = 0.6

    base_time = baseline * phase
    # Don't spend more than 50% of remaining time on one move
    max_time = 0.5 * time_remaining

    HARD_MIN = 0.02
    HARD_MAX = 5.0

    # Time trouble
    if time_remaining < 5.0:
        return min(time_remaining / 5, 0.5)

    move_time = max(HARD_MIN, min(base_time, HARD_MAX, max_time))

    return move_time

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
        new_node = Node(Colour.opposite(self.colour), move, self, new_board, self.root_colour)
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
    def search(self, limit):
        stop_time = time.time() + limit
        iterations = 0

        while time.time() < stop_time:
            node = self.select()
            if node.untried_moves:
                node = node.expand()
            iterations += 1

        best_child = max(self.children, key=lambda x: x.visits)
        return best_child.move

class BaseMCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        # Colour: red or blue - red moves first.
        super().__init__(colour)
        self.average_time = 0
        self.time_used = 0

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

        new_board = convert_bitboard(board)

        time_remaining = TIME_LIMIT - self.time_used

        root = Node(self.colour, opp_move, None, board, self.colour)
        move_limit = time_allocator(turn, new_board, time_remaining)

        start_time = time.time()
        response = root.search(move_limit)
        end_time = time.time()

        self.time_used += end_time - start_time

        if self.average_time == 0:
            self.average_time = end_time - start_time
        else:
            self.average_time = (self.average_time + (end_time - start_time)) / 2
        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        return response
