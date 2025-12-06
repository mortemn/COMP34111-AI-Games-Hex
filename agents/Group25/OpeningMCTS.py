from __future__ import annotations
import random
import subprocess
import time

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from agents.Group25.Bitboard import Bitboard, convert_bitboard
from agents.Group25.OpeningBook import OpeningBook
from src.Game import logger
from src.Tile import Tile
from math import sqrt, inf, log
from random import choice

C_EXPLORATION = 0.7
RAVE_CONSTANT = 700
RAVE_MAX_DEPTH = 7


# Measured in seconds, represents the maximum time allowed for whole game
TIME_LIMIT = 3 * 60

class Node:
    def __init__(self, colour: Colour, move: tuple[int, int] | None, parent, board: Bitboard, root_colour: Colour, depth: int = 0):
        # Colour to move in the current turn
        self.colour: Colour = colour
        # Last move that was made
        self.move: tuple[int, int] | None = move
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
        self.depth = depth

        self.rave_wins: dict[tuple[int, int], float] = {}
        self.rave_visits: dict[tuple[int, int], int] = {}

    def ucb(self, child: Node):
        # Explore unvisited children
        if child.visits == 0:
            return inf

        # UCT estimate with neutral smoothing
        q_uct = (child.wins + 1) / (child.visits + 2)

        # UCT exploration
        uct_exp = C_EXPLORATION * sqrt(log(self.visits + 1)/child.visits)

        # Explanation for child.move is None: currently in our implementation, there is no scenario where the root node is passed to ucb, but to be safe, this check is here
        if self.depth >= RAVE_MAX_DEPTH or child.move is None:
            # This is normal UCT without RAVE
            return q_uct + uct_exp

        # Calculate RAVE value
        rave_n = self.rave_visits.get(child.move, 0)
        rave_w = self.rave_wins.get(child.move, 0)

        # RAVE estimate with Bayesian smoothing
        q_rave = (rave_w + 1) / (rave_n + 2)

        # As depth increases, the influence of RAVE decreases
        beta = RAVE_CONSTANT / (RAVE_CONSTANT + child.visits)
        q_final = beta * q_rave + (1 - beta) * q_uct

        return q_final + uct_exp

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
        new_node = Node(Colour.opposite(self.colour), move, self, new_board, self.root_colour, self.depth + 1)
        self.children.append(new_node)
        return new_node

    def simulate(self):
        # While the game hasn't ended, keep on playing random moves 
        # TODO: Add a heuristic to select non-random moves

        new_board = self.board.copy()
        colour = self.colour

        trace: list[tuple[Colour, tuple[int, int]]] = []

        while True:
            moves = new_board.legal_moves()
            # Should not possibly happen but just in case
            if not moves:
                if new_board.red_won():
                    return Colour.RED, trace
                else:
                    return Colour.BLUE, trace

            idx = random.randrange(len(moves))
            x, y = moves.pop(idx)
            
            new_board.move_at(x, y, colour)
            trace.append((colour, (x, y)))

            # Only check for win from the previous player's color, reduces checks by half
            if colour == Colour.RED:
                if new_board.red_won():
                    return Colour.RED, trace
            else:
                if new_board.blue_won():
                    return Colour.BLUE, trace

            colour = Colour.opposite(colour)

    def backpropagate(self, winner: Colour, trace: list[tuple[Colour, tuple[int, int]]]):
        self.visits += 1 
        if winner == self.root_colour:
            self.wins += 1

        # Update RAVE stats
        for colour, move in trace:
            if colour == self.colour:
                # Update the number of visits for this move
                self.rave_visits[move] = self.rave_visits.get(move, 0) + 1
                wins_before = self.rave_wins.get(move, 0.0)
                # Win is updated depending on whether or not this move's colour is same as root
                if winner == self.root_colour:
                    self.rave_wins[move] = wins_before + 1.0
                else:
                    self.rave_wins[move] = wins_before

        if self.parent:
            self.parent.backpropagate(winner, trace)

    # TODO: make search time based to fit with time constraints of CW
    def search(self, limit):
        stop_time = time.time() + limit
        iterations = 0

        while time.time() < stop_time:
            node = self.select()
            if node.untried_moves:
                node = node.expand()
            winner, trace = node.simulate()
            node.backpropagate(winner, trace)
            iterations += 1

        best_child = max(self.children, key=lambda x: x.visits)
        return best_child, Move(best_child.move[0], best_child.move[1]), iterations

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        # Colour: red or blue - red moves first.
        super().__init__(colour)
        self.time_used = 0
        self.total_iterations = 0
        self.root = None
        self.opening_book = OpeningBook(colour)

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

        if self.opening_book.in_book(turn, opp_move):
            return self.opening_book.play_move(turn, opp_move)
            
        bitboard = convert_bitboard(board)
        if self.root is None:
            self.root = Node(self.colour, None, None, bitboard, self.colour)
        else:
            if opp_move is not None:
                if opp_move.x != -1 and opp_move.y != -1:
                    found = False
                    target = (opp_move.x, opp_move.y)
                    for child in self.root.children:
                        if child.move == target:
                            found = True
                            logger.debug("FOUND")
                            old_parent = child.parent
                            self.root = child
                            # Free references
                            if old_parent is not None:
                                old_parent.children = []
                                self.root.parent = None
                            self.root.board = bitboard
                            break
                    if found == False:
                        self.root = Node(self.colour, None, None, bitboard, self.colour)
                else:
                    # Pie rule used, easiest approach is to rebuild the Node, this can be improved
                    self.root = Node(self.colour, None, None, bitboard, self.colour)
            else:
                # Fallback condition, which probably also means we are red on the first move
                self.root = Node(self.colour, None, None, bitboard, self.colour)
                

        time_remaining = max(0.0, TIME_LIMIT - self.time_used)

        if time_remaining < 10:
            # If we're in time trouble, allocate at most 0.05 seconds per move
            if time_remaining <= 0.01:
                move_limit = 0.01
            else:
                move_limit = min(0.05, time_remaining)
        else:
            # Early game
            if turn < 10:
                base = 3.0
            # Mid game
            elif turn < 30:
                base = 2.0
            # End game
            else:
                base = 1.0

            # Conservation time usage
            move_limit = min(base, time_remaining / 2.0)


        start_time = time.time()
        best_child, response, iterations = self.root.search(move_limit)
        end_time = time.time()

        old_parent = best_child.parent
        self.root = best_child
        if old_parent is not None:
            old_parent.children = []
            self.root.parent = None

        time_spent = end_time - start_time
        self.time_used += time_spent
        self.total_iterations += iterations

        logger.log(10, f"RAVEMCTSAgent iterations per second: {self.total_iterations / self.time_used}")
        logger.log(10, f"RAVEMCTSAgent time used so far: {self.time_used}")

        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        return response
