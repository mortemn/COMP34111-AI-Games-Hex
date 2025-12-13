from __future__ import annotations
from multiprocessing import Pool, cpu_count
import time
import random

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from agents.Group25.Bitboard import convert_bitboard
from agents.Group25.EliminateCopy import (
    Node, find_forced_win, find_opp_forced_win, 
    time_allocator, TIME_LIMIT
)
from src.Game import logger

class ParallelMCTS(AgentBase):
    def __init__(self, colour: Colour, num_processes: int=8):
        super().__init__(colour)
        self.num_processes = num_processes
        self.time_used = 0
        self.total_iterations = 0

        logger.log(10, f"ParallelMCTS initialized with {self.num_processes} processes.")

    def make_move(self, turn, board, opp_move):
        time_remaining = max(0.0, TIME_LIMIT - self.time_used)
        bitboard = convert_bitboard(board)

        if len(bitboard.legal_moves()) <= 25 and time_remaining > 8.0:
            forced_win = find_forced_win(bitboard, self.colour)
            if forced_win is not None:
                return Move(forced_win[0], forced_win[1])

            threats = find_opp_forced_win(bitboard, self.colour)
            if len(threats) == 1:
                return Move(threats[0][0], threats[0][1])

        move_limit = time_allocator(turn, bitboard, time_remaining)

        start_time = time.time()

        with Pool(processes=self.num_processes) as pool:
            args = [
                (bitboard, self.colour, move_limit, turn * 1000 + i)
                for i in range(self.num_processes)
            ]
            
            results = pool.starmap(run_search, args)

        end_time = time.time()

        total_visits = {}
        total_iterations = 0

        for visits, iterations in results:
            total_iterations += iterations
            for move, count in visits.items():
                total_visits[move] = total_visits.get(move, 0) + count

        if not total_visits:
            # Fallback: play a random legal move
            logger.warning("No visits recorded, playing random move.")
            legal_moves = bitboard.legal_moves()
            move = random.choice(legal_moves) if legal_moves else (0, 0)
            return Move(move[0], move[1])

        best_move = max(total_visits.items(), key=lambda item: item[1])[0]

        time_spent = end_time - start_time
        self.time_used += time_spent
        self.total_iterations += total_iterations

        logger.log(10, f"ParallelMCTS iterations per second: {self.total_iterations / self.time_used}")
        logger.log(10, f"ParallelMCTS time used so far: {self.time_used}")

        return Move(best_move[0], best_move[1])


def run_search(bitboard, colour, move_limit, seed):
    random.seed(seed)

    root = Node(colour, None, None, bitboard, colour)
    root_depth = root.depth

    _, _, iterations = root.search(move_limit, root_depth)

    # Map moves to visit counts
    visits = {}
    for child in root.children:
        if child.move is not None:
            visits[child.move] = child.visits

    return visits, iterations
