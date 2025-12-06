# ==============================
#  Original RAVE MCTS Agent Code
#  (Provided by teammate, copied exactly)
#  + Root Parallelisation appended at bottom
# ==============================

from __future__ import annotations
import random
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

C_EXPLORATION = 0.7
RAVE_CONSTANT = 700
RAVE_MAX_DEPTH = 7

# Measured in seconds, represents the maximum time allowed for whole game
TIME_LIMIT = 3 * 60

class Node:
    def __init__(self, colour: Colour, move: tuple[int, int] | None, parent, board: Bitboard, root_colour: Colour, depth: int = 0):
        self.colour: Colour = colour
        self.move: tuple[int, int] | None = move
        self.parent = parent
        self.board = board
        self.visits = 0
        self.wins = 0
        self.children = []
        self.untried_moves = board.legal_moves()
        self.root_colour = root_colour
        self.depth = depth

        self.rave_wins: dict[tuple[int, int], float] = {}
        self.rave_visits: dict[tuple[int, int], int] = {}

    def ucb(self, child: "Node"):
        if child.visits == 0:
            return inf

        q_uct = (child.wins + 1) / (child.visits + 2)
        uct_exp = C_EXPLORATION * sqrt(log(self.visits + 1)/child.visits)

        if self.depth >= RAVE_MAX_DEPTH or child.move is None:
            return q_uct + uct_exp

        rave_n = self.rave_visits.get(child.move, 0)
        rave_w = self.rave_wins.get(child.move, 0)
        q_rave = (rave_w + 1) / (rave_n + 2)

        beta = RAVE_CONSTANT / (RAVE_CONSTANT + child.visits)
        q_final = beta * q_rave + (1 - beta) * q_uct

        return q_final + uct_exp

    def best_child(self):
        return max(self.children, key=lambda x: self.ucb(x))

    def select(self):
        node = self
        while not node.untried_moves and node.children:
            node = node.best_child()
        return node

    def expand(self):
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.move_at(move[0], move[1], self.colour)
        new_node = Node(Colour.opposite(self.colour), move, self, new_board, self.root_colour, self.depth + 1)
        self.children.append(new_node)
        return new_node

    def simulate(self):
        new_board = self.board.copy()
        colour = self.colour

        trace: list[tuple[Colour, tuple[int, int]]] = []

        while True:
            moves = new_board.legal_moves()
            if not moves:
                if new_board.red_won():
                    return Colour.RED, trace
                else:
                    return Colour.BLUE, trace

            idx = random.randrange(len(moves))
            x, y = moves.pop(idx)

            new_board.move_at(x, y, colour)
            trace.append((colour, (x, y)))

            if colour == Colour.RED:
                if new_board.red_won():
                    return Colour.RED, trace
            else:
                if new_board.blue_won():
                    return Colour.BLUE, trace

            colour = Colour.opposite(colour)

    def backpropagate(self, winner: Colour, trace: list[tuple[Colour, tuple[int, int]]] ):
        self.visits += 1
        if winner == self.root_colour:
            self.wins += 1

        for colour, move in trace:
            if colour == self.colour:
                self.rave_visits[move] = self.rave_visits.get(move, 0) + 1
                wins_before = self.rave_wins.get(move, 0.0)
                if winner == self.root_colour:
                    self.rave_wins[move] = wins_before + 1.0
                else:
                    self.rave_wins[move] = wins_before

        if self.parent:
            self.parent.backpropagate(winner, trace)

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
        super().__init__(colour)
        self.time_used = 0
        self.total_iterations = 0

    def make_move(self, turn: int, board, opp_move: Move | None) -> Move:
        if opp_move is not None:
            root = Node(self.colour, (opp_move.x, opp_move.y), None, convert_bitboard(board), self.colour)
        else:
            root = Node(self.colour, None, None, convert_bitboard(board), self.colour)

        time_remaining = max(0.0, TIME_LIMIT - self.time_used)

        if time_remaining < 10:
            move_limit = 0.05
        else:
            if turn < 10:
                move_limit = 3.0
            elif turn < 30:
                move_limit = 2.0
            else:
                move_limit = 1.0

        start = time.time()
        best_child, response, iterations = root.search(move_limit)
        end = time.time()

        self.time_used += (end - start)
        self.total_iterations += iterations

        logger.log(10, f"RAVEMCTSAgent iterations per second: {self.total_iterations / self.time_used}")
        logger.log(10, f"RAVEMCTSAgent time used so far: {self.time_used}")

        return response



# ===================================================================
#                ROOT PARALLEL MCTS ADDITION (STUDENT STYLE)
# ===================================================================

# Replace the previous RootParallelMCTSAgent with this improved version.
import multiprocessing
import os
from multiprocessing import Process, Manager
import traceback

class RootParallelMCTSAgent(AgentBase):
    """
    Root-parallel RAVE MCTS using multiprocessing for real CPU parallelism.
    Keeps a thread-based fallback if Bitboard/Node cannot be pickled.

    Student-style, simple and easy to follow.
    """

    def __init__(self, colour: Colour, num_workers: int = 4):
        super().__init__(colour)
        # sensible default, will be clamped at runtime
        self.requested_workers = num_workers
        self.time_used = 0
        self.total_iterations = 0

    def _proc_worker(self, board_bitboard, colour, opp_move, move_limit, return_list, idx):
        """
        Worker run in its own process. Builds its own Node tree and runs search.
        Puts a tuple (worker_stats_dict, iterations) into return_list at index idx.
        """
        try:
            # Give each worker a distinct RNG seed so playouts differ
            random.seed(time.time() + os.getpid())

            # Each worker creates its own root from the provided bitboard copy
            if opp_move is not None:
                root = Node(colour, (opp_move.x, opp_move.y), None, board_bitboard.copy(), colour)
            else:
                root = Node(colour, None, None, board_bitboard.copy(), colour)

            # Run MCTS for the allotted time
            _, _, iterations = root.search(move_limit)

            # Gather root children stats: move -> (wins, visits)
            stats = {}
            for child in root.children:
                if child.move is not None:
                    stats[child.move] = (child.wins, child.visits)

            return_list[idx] = (stats, iterations)
        except Exception:
            # Don't let a worker crash silently; return something visible.
            tb = traceback.format_exc()
            return_list[idx] = ({"__error__": tb}, 0)

    def _thread_worker_fallback(self, board_bitboard, colour, opp_move, move_limit, shared_list):
        """
        Fallback worker that runs in a thread (used only when multiprocessing fails).
        This is effectively the old behaviour and is retained as a safety net.
        """
        random.seed(time.time() + threading.get_ident())
        if opp_move is not None:
            root = Node(colour, (opp_move.x, opp_move.y), None, board_bitboard.copy(), colour)
        else:
            root = Node(colour, None, None, board_bitboard.copy(), colour)

        _, _, iterations = root.search(move_limit)
        stats = {}
        for child in root.children:
            if child.move is not None:
                stats[child.move] = (child.wins, child.visits)
        shared_list.append((stats, iterations))

    def make_move(self, turn: int, board, opp_move: Move | None) -> Move:
        # Convert engine board to internal bitboard once
        root_bb = convert_bitboard(board)

        # Compute sane worker count
        try:
            cpu_count = multiprocessing.cpu_count()
        except Exception:
            cpu_count = 2
        legal_moves = root_bb.legal_moves()
        n_legal = len(legal_moves) if legal_moves is not None else 0
        workers = max(1, min(self.requested_workers, cpu_count, n_legal or 1))

        # Time budget (same heuristic as original)
        time_remaining = max(0.0, TIME_LIMIT - self.time_used)
        if time_remaining < 10:
            move_limit = 0.05
        else:
            if turn < 10:
                move_limit = 3.0
            elif turn < 30:
                move_limit = 2.0
            else:
                move_limit = 1.0

        start = time.time()

        # We'll try multiprocessing first (gives true parallelism on Linux/Docker).
        use_processes = True
        mgr = Manager()
        results = mgr.list([None] * workers)

        procs = []
        try:
            for i in range(workers):
                # pass a copy of the bitboard so each process has its own root
                bcopy = root_bb.copy()
                p = Process(target=self._proc_worker, args=(bcopy, self.colour, opp_move, move_limit, results, i))
                p.start()
                procs.append(p)

            # wait for all processes
            for p in procs:
                p.join()

        except Exception:
            # If anything goes wrong (e.g., Bitboard not picklable), fallback to threads
            use_processes = False
            # terminate any started processes to be safe
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass

            # thread fallback (keeps older behaviour)
            import threading as _threading
            thread_results = []
            threads = []
            for i in range(workers):
                bcopy = root_bb.copy()
                t = _threading.Thread(target=self._thread_worker_fallback,
                                      args=(bcopy, self.colour, opp_move, move_limit, thread_results))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            # Normalize to same structure as results (list of length workers)
            results = thread_results

        end = time.time()

        # Aggregate stats
        merged = {}
        total_iterations = 0

        if use_processes:
            for entry in results:
                if not entry:
                    continue
                worker_stats, iters = entry
                # detect worker exception
                if isinstance(worker_stats, dict) and "__error__" in worker_stats:
                    logger.log(1, f"Worker error: {worker_stats['__error__']}")
                    continue
                total_iterations += iters
                for move, (w, v) in worker_stats.items():
                    if move not in merged:
                        merged[move] = [0.0, 0]
                    merged[move][0] += w
                    merged[move][1] += v
        else:
            # thread fallback: 
            for worker_stats, iters in results:
                total_iterations += iters
                for move, (w, v) in worker_stats.items():
                    if move not in merged:
                        merged[move] = [0.0, 0]
                    merged[move][0] += w
                    merged[move][1] += v

        # Choose best move by total visits, tiebreaker wins
        best_move = None
        best_visits = -1
        best_wins = -1.0
        for move, (w, v) in merged.items():
            if v > best_visits or (v == best_visits and w > best_wins):
                best_move = move
                best_visits = v
                best_wins = w

        # fallback to random legal if nothing found
        if best_move is None:
            legals = root_bb.legal_moves()
            if not legals:
                return Move(-1, -1)
            m = choice(legals)
            selected = Move(m[0], m[1])
        else:
            selected = Move(best_move[0], best_move[1])

        self.total_iterations += total_iterations
        self.time_used += (time.time() - start)

        logger.log(10, f"RootParallelRAVE workers={workers} processes={use_processes} iters={self.total_iterations} time_used={self.time_used:.3f}")
        return selected
