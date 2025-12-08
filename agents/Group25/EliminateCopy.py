from __future__ import annotations
import random
import subprocess
import time
from collections import defaultdict

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from agents.Group25.Bitboard import Bitboard, convert_bitboard
from agents.Group25.OpeningBook import OpeningBook
from src.Game import logger
from src.Tile import Tile
from math import sqrt, inf, log
from random import choice

# CONSTANTS

# RAVE
C_EXPLORATION = 0.7
RAVE_CONSTANT = 700
RAVE_MAX_DEPTH = 7

# Ordering
MAX_ORDER_DEPTH = 1
LOCALITY_D1 = 10
LOCALITY_D2 = 5

# Measured in seconds, represents the maximum time allowed for whole game
TIME_LIMIT = 3 * 60

NEIGBOUR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1),  (1, 0), (1, 1)]

# Moves that are closed to opponent's last move are prioritised
def locality(opp_move: tuple[int, int], candidate_move: tuple[int, int]):
    dx = abs(opp_move[0] - candidate_move[0])
    dy = abs(opp_move[1] - candidate_move[1])

    if (dx, dy) in [(0, 1), (1, 0), (1, 1)]:
        return LOCALITY_D1
    elif max(dx, dy) <= 2:
        return LOCALITY_D2

    return 0

# Moves that neighbour other stones are prioritised
def adjacency(board: Bitboard, candidate_colour: Colour, candidate_move: tuple[int, int]):
    x, y = candidate_move
    score = 0

    for dx, dy in NEIGBOUR_OFFSETS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < board.size and 0 <= ny < board.size:
            if board.colour_at(nx, ny) == candidate_colour:
                score += 3
            elif board.colour_at(nx, ny) == Colour.opposite(candidate_colour):
                score += 2

    return score

# Moves closer to the centre are prioritised
def centrality(board: Bitboard, candidate_move: tuple[int, int]):
    center = (board.size - 1) / 2
    x, y = candidate_move
    dist = abs(x - center) + abs(y - center)
    return -int(dist)

def score_move(board: Bitboard, colour: Colour, candidate_move: tuple[int, int], opp_move: tuple[int, int] | None):
    score = 0

    if opp_move is not None:
        score += locality(opp_move, candidate_move)

    score += adjacency(board, colour, candidate_move)
    score += centrality(board, candidate_move)

    return score

def find_forced_win(board: Bitboard, colour: Colour):
    for move in board.legal_moves():
        board.move_at(move[0], move[1], colour)
        
        won = (colour == Colour.RED and board.red_won()) or (colour == Colour.BLUE and board.blue_won())

        board.undo_at(move[0], move[1], colour)

        if won:
            return move
        
    return None

def find_opp_forced_win(board: Bitboard, colour: Colour):
    opp_colour = Colour.opposite(colour)
    wins = []

    for move in board.legal_moves():
        board.move_at(move[0], move[1], opp_colour)
        
        if opp_colour == Colour.RED and board.red_won():
            wins.append(move)
        elif opp_colour == Colour.BLUE and board.blue_won():
            wins.append(move)
        
        board.undo_at(move[0], move[1], opp_colour)

    return wins

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
        self.root_colour = root_colour
        self.depth = depth
        moves = board.legal_moves()
        if not moves:
            self.untried_moves = []
        else:
            if depth <= MAX_ORDER_DEPTH:
                scored_moves = []
                for move in moves:
                    score = score_move(board, colour, move, self.move)
                    # Add a random tiebreaker
                    scored_moves.append((score, random.random(), move))
                scored_moves.sort(reverse=True, key=lambda x: (x[0], x[1]))
                self.untried_moves = [move for _, _, move in scored_moves]
            else:
                random.shuffle(moves)
                self.untried_moves = moves

        self.rave_wins: dict[tuple[int, int], float] = defaultdict(float)
        self.rave_visits: dict[tuple[int, int], int] = defaultdict(int)

    def ucb(self, child: Node, root_depth: int):
        # Explore unvisited children
        if child.visits == 0:
            return inf

        # UCT estimate with neutral smoothing
        q_uct = (child.wins + 1) / (child.visits + 2)

        # UCT exploration
        uct_exp = C_EXPLORATION * sqrt(log(self.visits + 1)/child.visits)

        # Explanation for child.move is None: currently in our implementation, there is no scenario where the root node is passed to ucb, but to be safe, this check is here
        effective_depth = self.depth - root_depth
        if effective_depth >= RAVE_MAX_DEPTH or child.move is None:
            # This is normal UCT without RAVE
            return q_uct + uct_exp

        # Calculate RAVE value
        rave_n = self.rave_visits[child.move]
        rave_w = self.rave_wins[child.move]

        # RAVE estimate with Bayesian smoothing
        q_rave = (rave_w + 1) / (rave_n + 2)

        # As depth increases, the influence of RAVE decreases
        beta = RAVE_CONSTANT / (RAVE_CONSTANT + child.visits)
        q_final = beta * q_rave + (1 - beta) * q_uct

        return q_final + uct_exp

    def best_child(self, root_depth: int):
        return max(self.children, key=lambda x: self.ucb(x, root_depth))

    def select(self, root_depth):
        # TODO: What if there are multiple children of the same UCB? The selection might not be truly random. It might also be worth it to implement a heuristic for this
        
        # If there are no untried moves, iterate until a child with untried moves is reached
        node = self
        while not node.untried_moves and node.children:
            node = node.best_child(root_depth)
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

        new_board = self.board
        colour = self.colour
        size = new_board.size

        trace: list[tuple[Colour, tuple[int, int]]] = []
        move_stack = []
        playout_depth = 0
        last_move = self.move

        legal_moves = new_board.legal_moves()
        seed = random.randrange(9999999999999999)

        try:
            while legal_moves:
                if playout_depth <= 15:
                    LOCAL_PROB = 30
                    # ADJ_PROB = 0.1
                elif playout_depth <= 35:
                    LOCAL_PROB = 60
                    # ADJ_PROB = 0.2
                else:
                    LOCAL_PROB = 25
                    # ADJ_PROB = 0.1

                move = None

                # Somewhat random
                use_local = (hash((seed, playout_depth)) % 100) < LOCAL_PROB

                # Local move heuristic
                if last_move is not None and use_local:
                    lx, ly = last_move
                    local = []
                    for dx, dy in NEIGBOUR_OFFSETS:
                        nx, ny = lx + dx, ly + dy
                        if 0 <= nx < size and 0 <= ny < size and (nx, ny) in legal_moves:
                            local.append((nx, ny))
                    if local:
                        index = hash((seed, playout_depth, lx, ly)) % len(local)
                        move = local[index]
                        legal_moves.remove(move)

                
                # Adjacency heuristic
                # if move is None and random.random() < ADJ_PROB:
                #     adjacent = []
                #     for candidate in moves:
                #         for dx, dy in NEIGBOUR_OFFSETS:
                #             nx, ny = candidate[0] + dx, candidate[1] + dy
                #             if 0 <= nx < size and 0 <= ny < size:
                #                 if new_board.colour_at(nx, ny) == colour:
                #                     adjacent.append(candidate)
                #                     print(candidate, (nx, ny))
                #                     break
                #     if adjacent:
                #         move = random.choice(adjacent)

                if move is None:
                    index = hash((seed, playout_depth)) % len(legal_moves)
                    move = legal_moves.pop(index)
                
                new_board.move_at(move[0], move[1], colour)
                move_stack.append((move, colour))
                trace.append((colour, (move[0], move[1])))

                # Only check for win from the previous player's color, reduces checks by half
                if colour == Colour.RED:
                    if new_board.red_can_win() and new_board.red_won():
                        return Colour.RED, trace
                else:
                    if new_board.blue_can_win() and new_board.blue_won():
                        return Colour.BLUE, trace

                last_move = move
                playout_depth += 1
                colour = Colour.opposite(colour)
            if new_board.red_won():
                return Colour.RED, trace
            else:
                return Colour.BLUE, trace
        finally:
            for move, colour in reversed(move_stack):
                new_board.undo_at(move[0], move[1], colour)

    def backpropagate(self, winner: Colour, trace: list[tuple[Colour, tuple[int, int]]]):
        self.visits += 1 
        if winner == self.root_colour:
            self.wins += 1

        # Update RAVE stats
        for colour, move in trace:
            if colour == self.colour:
                # Update the number of visits for this move
                self.rave_visits[move] += 1
                # Win is updated depending on whether or not this move's colour is same as root
                if winner == self.root_colour:
                    self.rave_wins[move] += 1.0

        if self.parent:
            self.parent.backpropagate(winner, trace)

    def search(self, limit, root_depth: int):
        stop_time = time.time() + limit
        iterations = 0

        while time.time() < stop_time:
            node = self.select(root_depth)
            if node.untried_moves:
                node = node.expand()
            winner, trace = node.simulate()
            node.backpropagate(winner, trace)
            iterations += 1

        best_child = max(self.children, key=lambda x: x.visits)
        return best_child, Move(best_child.move[0], best_child.move[1]), iterations

class EliminateCopyMCTS(AgentBase):
    def __init__(self, colour: Colour):
        # Colour: red or blue - red moves first.
        super().__init__(colour)
        self.time_used = 0
        self.total_iterations = 0
        self.root = None
        self.root_depth = 0
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

        time_remaining = max(0.0, TIME_LIMIT - self.time_used)
        bitboard = convert_bitboard(board)

        if self.opening_book.in_book(turn, opp_move):
            return self.opening_book.play_move(turn, opp_move)

        if len(bitboard.legal_moves()) <= 25 and time_remaining > 8.0:
            forced_win = find_forced_win(bitboard, self.colour)
            if forced_win is not None:
                return Move(forced_win[0], forced_win[1])

            threats = find_opp_forced_win(bitboard, self.colour)
            if len(threats) == 1:
                return Move(threats[0][0], threats[0][1])

        # If len(threats) > 1, cry
            
        if self.root is None:
            self.root = Node(self.colour, None, None, bitboard, self.colour)
            self.root_depth = self.root.depth
        else:
            if opp_move is not None:
                if opp_move.x != -1 and opp_move.y != -1:
                    found = False
                    target = (opp_move.x, opp_move.y)
                    for child in self.root.children:
                        if child.move == target:
                            found = True
                            old_parent = child.parent
                            self.root = child
                            self.root_depth = self.root.depth
                            # Free references
                            if old_parent is not None:
                                old_parent.children = []
                                self.root.parent = None
                            self.root.board = bitboard
                            break
                    if found == False:
                        self.root = Node(self.colour, None, None, bitboard, self.colour)
                        self.root_depth = self.root.depth
                else:
                    # Pie rule used, easiest approach is to rebuild the Node, this can be improved
                    self.root = Node(self.colour, None, None, bitboard, self.colour)
                    self.root_depth = self.root.depth
            else:
                # Fallback condition, which probably also means we are red on the first move
                self.root = Node(self.colour, None, None, bitboard, self.colour)
                self.root_depth = self.root.depth

        move_limit = time_allocator(turn, bitboard, time_remaining)

        start_time = time.time()
        best_child, response, iterations = self.root.search(move_limit, self.root_depth)
        end_time = time.time()

        old_parent = best_child.parent
        self.root = best_child
        if old_parent is not None:
            old_parent.children = []
            self.root.parent = None
        self.root_depth = self.root.depth

        time_spent = end_time - start_time
        self.time_used += time_spent
        self.total_iterations += iterations

        logger.log(10, f"RAVEMCTSAgent iterations per second: {self.total_iterations / self.time_used}")
        logger.log(10, f"RAVEMCTSAgent time used so far: {self.time_used}")

        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        return response
