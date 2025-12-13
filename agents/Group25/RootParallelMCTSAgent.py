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

import multiprocessing
from multiprocessing import Process, Manager
import os
import traceback
import threading


C_EXPLORATION = 0.7
RAVE_CONSTANT = 700
RAVE_MAX_DEPTH = 7
TIME_LIMIT = 3 * 60


class Node:
    def __init__(self, colour: Colour, move: tuple[int, int] | None, parent, board: Bitboard, root_colour: Colour, depth: int = 0):
        self.colour = colour
        self.move = move
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


    def ucb(self, child):
        if child.visits == 0:
            return inf

        q_uct = (child.wins + 1) / (child.visits + 2)
        uct_exp = C_EXPLORATION * sqrt(log(self.visits + 1) / child.visits)

        if self.depth >= RAVE_MAX_DEPTH or child.move is None:
            return q_uct + uct_exp

        rave_n = self.rave_visits.get(child.move, 0)
        rave_w = self.rave_wins.get(child.move, 0)
        q_rave = (rave_w + 1) / (rave_n + 2)

        beta = RAVE_CONSTANT / (RAVE_CONSTANT + child.visits)
        q = beta * q_rave + (1 - beta) * q_uct

        return q + uct_exp


    def best_child(self):
        return max(self.children, key=lambda c: self.ucb(c))


    def select(self):
        node = self
        while not node.untried_moves and node.children:
            node = node.best_child()
        return node


    def expand(self):
        move = self.untried_moves.pop()
        b2 = self.board.copy()
        b2.move_at(move[0], move[1], self.colour)
        new_node = Node(Colour.opposite(self.colour), move, self, b2, self.root_colour, self.depth + 1)
        self.children.append(new_node)
        return new_node


    def simulate(self):
        b2 = self.board.copy()
        colour = self.colour
        trace = []

        while True:
            moves = b2.legal_moves()
            if not moves:
                if b2.red_won():
                    return Colour.RED, trace
                return Colour.BLUE, trace

            m = random.choice(moves)
            x, y = m
            b2.move_at(x, y, colour)
            trace.append((colour, (x, y)))

            if colour == Colour.RED:
                if b2.red_won():
                    return Colour.RED, trace
            else:
                if b2.blue_won():
                    return Colour.BLUE, trace

            colour = Colour.opposite(colour)


    def backpropagate(self, winner, trace):
        self.visits += 1
        if winner == self.root_colour:
            self.wins += 1

        for col, move in trace:
            if col == self.colour:
                self.rave_visits[move] = self.rave_visits.get(move, 0) + 1
                if winner == self.root_colour:
                    self.rave_wins[move] = self.rave_wins.get(move, 0) + 1

        if self.parent:
            self.parent.backpropagate(winner, trace)


    def search(self, limit):
        end_time = time.time() + limit
        iterations = 0

        while time.time() < end_time:
            node = self.select()
            if node.untried_moves:
                node = node.expand()

            winner, trace = node.simulate()
            node.backpropagate(winner, trace)
            iterations += 1

        best = max(self.children, key=lambda c: c.visits)
        return best, Move(best.move[0], best.move[1]), iterations



class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.time_used = 0
        self.total_iterations = 0


    def make_move(self, turn, board, opp_move):
        bb = convert_bitboard(board)

        if opp_move is not None:
            root = Node(self.colour, (opp_move.x, opp_move.y), None, bb, self.colour)
        else:
            root = Node(self.colour, None, None, bb, self.colour)

        time_remaining = max(0, TIME_LIMIT - self.time_used)

        if time_remaining < 10:
            limit = 0.05
        elif turn < 10:
            limit = 3.0
        elif turn < 30:
            limit = 2.0
        else:
            limit = 1.0

        start = time.time()
        child, move, iters = root.search(limit)
        self.time_used += time.time() - start
        self.total_iterations += iters

        return move



class RootParallelMCTSAgent(AgentBase):
    def __init__(self, colour: Colour, num_workers: int = 4):
        super().__init__(colour)
        self.requested_workers = num_workers
        self.time_used = 0
        self.total_iterations = 0


    def _proc_worker(self, board_bitboard, colour, opp_move, limit, out_list, idx):
        try:
            random.seed(time.time() + os.getpid())

            bcopy = board_bitboard.copy()

            if opp_move:
                root = Node(colour, (opp_move.x, opp_move.y), None, bcopy, colour)
            else:
                root = Node(colour, None, None, bcopy, colour)

            _, _, iters = root.search(limit)

            stats = {}
            for c in root.children:
                if c.move is not None:
                    stats[c.move] = (c.wins, c.visits)

            out_list[idx] = (stats, iters)

        except Exception:
            out_list[idx] = ({"_error_": traceback.format_exc()}, 0)


    def _thread_worker_fallback(self, board_bb, colour, opp_move, limit, shared):
        random.seed(time.time() + threading.get_ident())
        bcopy = board_bb.copy()

        if opp_move:
            root = Node(colour, (opp_move.x, opp_move.y), None, bcopy, colour)
        else:
            root = Node(colour, None, None, bcopy, colour)

        _, _, iters = root.search(limit)

        stats = {}
        for c in root.children:
            if c.move:
                stats[c.move] = (c.wins, c.visits)

        shared.append((stats, iters))


    def make_move(self, turn, board, opp_move):
        bb = convert_bitboard(board).copy()

        # Determine worker count
        try:
            cpu = multiprocessing.cpu_count()
        except:
            cpu = 2

        legal = bb.legal_moves()
        n_legal = len(legal) if legal else 1

        workers = max(1, min(self.requested_workers, cpu, n_legal))

        # Time budget
        rem = max(0, TIME_LIMIT - self.time_used)
        if rem < 10:
            limit = 0.05
        elif turn < 10:
            limit = 3.0
        elif turn < 30:
            limit = 2.0
        else:
            limit = 1.0

        start_time = time.time()

        manager = Manager()
        results = manager.list([None] * workers)
        procs = []
        use_processes = True

        try:
            for i in range(workers):
                bcopy = bb.copy()
                p = Process(target=self._proc_worker, args=(bcopy, self.colour, opp_move, limit, results, i))
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

        except Exception:
            # fallback to threads
            use_processes = False
            for p in procs:
                try:
                    p.terminate()
                except:
                    pass

            threads = []
            thread_results = []
            for i in range(workers):
                bcopy = bb.copy()
                t = threading.Thread(target=self._thread_worker_fallback,
                                     args=(bcopy, self.colour, opp_move, limit, thread_results))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            results = thread_results


        # Merge statistics
        merged = {}
        total_iters = 0

        for entry in results:
            if not entry:
                continue

            stats, iters = entry
            total_iters += iters

            if "_error_" in stats:
                logger.log(1, f"Worker error: {stats['_error_']}")
                continue

            for mv, (w, v) in stats.items():
                if mv not in merged:
                    merged[mv] = [0.0, 0]
                merged[mv][0] += w
                merged[mv][1] += v

        # Pick best move
        best_move = None
        best_visits = -1

        for mv, (w, v) in merged.items():
            if v > best_visits:
                best_move = mv
                best_visits = v

        if best_move is None:
            m = random.choice(bb.legal_moves())
            result = Move(m[0], m[1])
        else:
            result = Move(best_move[0], best_move[1])

        self.total_iterations += total_iters
        self.time_used += time.time() - start_time

        return result
