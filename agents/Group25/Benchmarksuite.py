import random
from collections import defaultdict
import time
from src.Board import Board
from src.Colour import Colour

def apply_move_to_board(board: Board, move, colour: Colour):
    if move is None:
        raise ValueError("Agent returned None move")

    x, y = move.x, move.y
    if hasattr(board, "move_at"):
        board.move_at(x, y, colour)
        return
    if hasattr(board, "apply_move"):
        try:
            board.apply_move(x, y, colour)
        except TypeError:
            board.apply_move(move, colour)
        return
    if hasattr(board, "make_move"):
        board.make_move(x, y, colour)
        return
    if hasattr(board, "set_tile"):
        board.set_tile(x, y, colour)
        return

    raise AttributeError("Board does not expose a method to apply moves.")


def play_single_game(agent1, agent2, board_size = 11, verbose: bool = False):
    board = Board(board_size)

    if isinstance(agent1, type):
        a1 = agent1(Colour.RED)
    else:
        a1 = agent1
    if isinstance(agent2, type):
        a2 = agent2(Colour.BLUE)
    else:
        a2 = agent2
    
    current_agent = a1
    other_agent = a2
    opp_move = None
    turn = 1

    max_turns = board_size * board_size + 5

    while True:
        start = time.time()
        move = current_agent.make_move(turn, board, opp_move)
        elapsed = time.time() - start

        if verbose:
            print(f"[turn {turn}] {current_agent.__class__.__name__} -> {getattr(move, 'x', None)},{getattr(move,'y',None)} (took {elapsed:.3f}s)")

        try:
            apply_move_to_board(board, move, current_agent.colour)
        except Exception as e:
            if verbose:
                print("Error applying move:", e)
            # if current_agent is red and illegal, blue wins; else red wins
            return "BLUE" if current_agent.colour == Colour.RED else "RED"

        # Check win condition using board methods
        if hasattr(board, "red_won") and board.red_won():
            return "RED"
        if hasattr(board, "blue_won") and board.blue_won():
            return "BLUE"

        # swap agents for next turn
        opp_move = move
        current_agent, other_agent = other_agent, current_agent
        turn += 1

        if turn > max_turns:
            red_count = 0
            blue_count = 0
            if hasattr(board, "tiles"):
                try:
                    for row in board.tiles:
                        for tile in row:
                            if getattr(tile, "colour", None) is None:
                                continue
                            if tile.colour == Colour.RED:
                                red_count += 1
                            else:
                                blue_count += 1
                except Exception:
                    pass
            # choose winner
            if red_count >= blue_count:
                return "RED"
            else:
                return "BLUE"


def run_matches(agent1_class, agent2_class, games, board_size=11):
    """
    Plays many games (games count) between agent1 and agent2.
    agent1_class and agent2_class can be classes or callables that produce agents.
    Returns win stats.
    """
    results = {"agent1": 0, "agent2": 0}

    for i in range(games):
        winner = play_single_game(agent1_class, agent2_class, board_size=board_size, verbose=False)
        if winner == "RED":
            results["agent1"] += 1
        else:
            results["agent2"] += 1

    return results

# ELO SYSTEM
def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def update_elo(r_a, r_b, score_a, k=32):
    e_a = expected_score(r_a, r_b)
    new_a = r_a + k * (score_a - e_a)
    new_b = r_b + k * ((1 - score_a) - (1 - e_a))
    return new_a, new_b


def round_robin_elos(agent_classes, games_each, board_size=11):
    """
    Performs a complete round robin:
    Every agent plays every other agent (as Red)
    and then the reverse matchup (as Blue).

    agent_classes: list of (name, class)
    games_each: how many games per matchup
    Returns dictionary: agent_name -> final Elo and win matrix
    """

    # Start all agents at Elo 1200
    elos = {name: 1200.0 for name, _ in agent_classes}

    # Record head-to-head results
    results_matrix = defaultdict(lambda: defaultdict(int))

    for i in range(len(agent_classes)):
        for j in range(i + 1, len(agent_classes)):
            name_a, class_a = agent_classes[i]
            name_b, class_b = agent_classes[j]

            # A as Red, B as Blue
            res1 = run_matches(class_a, class_b, games_each, board_size=board_size)
            wins_a = res1["agent1"]
            wins_b = res1["agent2"]

            # Update Elo based on average score
            total = wins_a + wins_b
            if total == 0:
                continue
            score_a = wins_a / total

            elos[name_a], elos[name_b] = update_elo(elos[name_a], elos[name_b], score_a)

            results_matrix[name_a][name_b] += wins_a
            results_matrix[name_b][name_a] += wins_b

            # Reverse matchup: B as Red, A as Blue
            res2 = run_matches(class_b, class_a, games_each, board_size=board_size)
            wins_b2 = res2["agent1"]
            wins_a2 = res2["agent2"]

            total2 = wins_a2 + wins_b2
            if total2 == 0:
                continue
            score_a2 = wins_a2 / total2

            elos[name_a], elos[name_b] = update_elo(elos[name_a], elos[name_b], score_a2)

            results_matrix[name_a][name_b] += wins_a2
            results_matrix[name_b][name_a] += wins_b2

    return elos, results_matrix
