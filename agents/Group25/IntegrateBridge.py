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

C_EXPLORATION = 0.7
RAVE_CONSTANT = 700
RAVE_MAX_DEPTH = 7


# Measured in seconds, represents the maximum time allowed for whole game
TIME_LIMIT = 3 * 60

BRIDGE_COMPLETE_SCORE = 3
BRIDGE_POTENTIAL_SCORE = 1

BRIDGE_NEIGHBOURS = [
    (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0)
]

# Cache: size -> { (cut_x,cut_y) : [ (other_cut, endpoint1, endpoint2), ... ] }
BRIDGE_PATTERNS_CACHE: dict[int, dict[tuple[int, int],
                                      list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]] = {}

def in_bounds(x: int, y: int, size: int) -> bool:
    """Check if (x, y) lies inside the board."""
    return 0 <= x < size and 0 <= y < size


def neighbours_of(x: int, y: int, size: int) -> set[tuple[int, int]]:
    """
    Return the set of neighbour coordinates of (x, y)
    under the Hex adjacency (6 neighbours max).
    """
    result: set[tuple[int, int]] = set()
    for dx, dy in BRIDGE_NEIGHBOURS:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, size):
            result.add((nx, ny))
    return result


def precompute_bridge_patterns(size: int) -> dict[tuple[int, int],
                                                  list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]:
    """
    Precompute all bridge patterns for a given board size.

    A bridge is defined as:
      - two endpoint cells A, B
      - exactly two shared neighbours C, D (the "cut" cells)

    We store, for each cut cell C:
      C -> list of (D, A, B) tuples

    and symmetrically for D.
    """
    patterns: dict[tuple[int, int], list[tuple[tuple[int, int],
                                               tuple[int, int], tuple[int, int]]]] = defaultdict(list)

    coords: list[tuple[int, int]] = [(x, y)
                                     for x in range(size) for y in range(size)]
    n = len(coords)

    for i in range(n):
        ax, ay = coords[i]
        neigh_a = neighbours_of(ax, ay, size)
        for j in range(i + 1, n):
            bx, by = coords[j]
            neigh_b = neighbours_of(bx, by, size)
            shared = neigh_a & neigh_b

            # A bridge pattern has exactly two shared neighbours
            if len(shared) == 2:
                c, d = tuple(shared)
                # c and d are the cut points; (ax,ay) and (bx,by) are endpoints
                patterns[c].append((d, (ax, ay), (bx, by)))
                patterns[d].append((c, (ax, ay), (bx, by)))

    return patterns


def get_bridge_patterns_for_size(size: int) -> dict[tuple[int, int],
                                                    list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]:
    """Get (and cache) bridge patterns for this board size."""
    if size not in BRIDGE_PATTERNS_CACHE:
        BRIDGE_PATTERNS_CACHE[size] = precompute_bridge_patterns(size)
    return BRIDGE_PATTERNS_CACHE[size]


def bridge_score(board: Bitboard, colour: Colour, move: tuple[int, int]) -> int:
    """
    Simple heuristic score for a move based on bridge patterns.

    We only score moves that are one of the "cut" cells in a bridge pattern.

    Heuristic:
      - BRIDGE_COMPLETE_SCORE if this move is a cut cell between two of our stones and
        the other cut cell is empty/ours (completes / strengthens a robust bridge).
      - BRIDGE_POTENTIAL_SCORE if this move is a cut cell where exactly one endpoint is ours
        and the other endpoint is empty, and the other cut is empty/ours
        (helps form a future bridge).
    """
    size = board.size
    patterns = get_bridge_patterns_for_size(size)

    mx, my = move
    candidates = patterns.get((mx, my), [])
    if not candidates:
        return 0

    score = 0
    for other_cut, ep1, ep2 in candidates:
        ex1, ey1 = ep1
        ex2, ey2 = ep2
        ox, oy = other_cut

        c1 = board.colour_at(ex1, ey1)
        c2 = board.colour_at(ex2, ey2)
        other_c = board.colour_at(ox, oy)

        # Treat other cut as "not blocked" if empty or ours
        other_ok = (other_c is None) or (other_c == colour)
        if not other_ok:
            continue

        # Both endpoints already ours: completing/strengthening a bridge
        if c1 == colour and c2 == colour:
            score += BRIDGE_COMPLETE_SCORE
        # One endpoint ours, the other empty: building a potential bridge
        elif (c1 == colour and c2 is None) or (c2 == colour and c1 is None):
            score += BRIDGE_POTENTIAL_SCORE

    return score

def find_forced_win(board: Bitboard, colour: Colour):
    for move in board.legal_moves():
        board.move_at(move[0], move[1], colour)
        
        if colour == Colour.RED and board.red_won():
            board.undo_at(move[0], move[1], colour)
            return move
        elif colour == Colour.BLUE and board.blue_won():
            board.undo_at(move[0], move[1], colour)
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
        moves = board.legal_moves()
        if not moves:
            self.untried_moves = []
        else:
            scored = [
                (bridge_score(board, self.colour, m), random.random(), m)
                for m in moves
            ]
            # Sort by (bridge_score, random_tiebreak) descending
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            self.untried_moves = [m for _, _, m in scored]

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

        time_remaining = max(0.0, TIME_LIMIT - self.time_used)
        bitboard = convert_bitboard(board)

        if self.opening_book.in_book(turn, opp_move):
            return self.opening_book.play_move(turn, opp_move)

        if len(bitboard.legal_moves()) <= 25 or time_remaining > 8.0:
            forced_win = find_forced_win(convert_bitboard(board), self.colour)
            if forced_win is not None:
                return Move(forced_win[0], forced_win[1])

            threats = find_opp_forced_win(convert_bitboard(board), self.colour)
            if len(threats) == 1:
                return Move(threats[0][0], threats[0][1])

        # If len(threats) > 1, cry
            
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
