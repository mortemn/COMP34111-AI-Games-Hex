from __future__ import annotations
import subprocess
import time

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from agents.Group25.Bitboard import Bitboard, convert_bitboard
from src.Game import logger
from src.Tile import Tile
from math import sqrt, inf, log
from random import choice

class Node:
    def __init__(self, colour: Colour, move: Move, parent, board: Bitboard, root_colour: Colour):
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
        self.untried_moves = board.legal_moves()
        self.root_colour = root_colour
        # Exploration parameter - can be tuned
        self.c = sqrt(2)

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
        new_board = self.board.copy()
        # Make move on board
        new_board.move_at(move[0], move[1], self.colour) 
        new_node = Node(Colour.opposite(self.colour), move, self, new_board, self.root_colour)
        self.children.append(new_node)
        return new_node

    def simulate(self):
        # While the game hasn't ended, keep on playing random moves 
        # TODO: Add a heuristic to select non-random moves

        new_board = self.board.copy()
        colour = self.colour

        while True:
            if new_board.red_won():
                return Colour.RED
            if new_board.blue_won():
                return Colour.BLUE

            moves = new_board.legal_moves()
            
            move = choice(moves)
            new_board.move_at(move[0], move[1], colour)
            colour = Colour.opposite(colour)

    def backpropagate(self, winner: Colour):
        self.visits += 1 
        if winner == self.root_colour:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(winner)

    # TODO: make search time based to fit with time constraints of CW
    def search(self, iterations):
        for _ in range(iterations):
            node = self.select()
            if node.untried_moves:
                node = node.expand()
            winner = node.simulate()
            node.backpropagate(winner)

        best_child = max(self.children, key=lambda x: x.visits)
        return Move(best_child.move[0], best_child.move[1])

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        # Colour: red or blue - red moves first.
        super().__init__(colour)
        self.average_time = 0

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

        # Convert board to internal representation
        # for row in board.tiles:
            

        root = Node(self.colour, opp_move, None, convert_bitboard(board), self.colour)
        iterations = 50
        start_time = time.time()
        response = root.search(iterations)
        end_time = time.time()
        if self.average_time == 0:
            self.average_time = end_time - start_time
        else:
            self.average_time = (self.average_time + (end_time - start_time)) / 2
        logger.info(f"Bitboard MCTS average time per move ({iterations} iterations): {self.average_time} seconds")

        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        return response
