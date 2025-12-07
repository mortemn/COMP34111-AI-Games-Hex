import time
from collections import defaultdict

from src.Colour import Colour
from src.Move import Move
from src.Board import Board

def _apply_move_board(board: Board, move: Move, colour: Colour):
    if move is None:
        raise ValueError("Agent returned None move")

    x, y = move.x, move.y

    if x == -1 and y == -1:
        if hasattr(board, "swap_sides"):
            board.swap_sides()
            return
        raise ValueError("Swap requested but board.swap_sides() unavailable")

    if not (0 <= x < board.size and 0 <= y < board.size):
        raise ValueError(f"Move out of bounds: {(x,y)}")

    # Ensure tile is empty before applying
    tile = board.tiles[x][y]
    if tile.colour is not None:
        raise ValueError(f"Tile already occupied at {(x,y)}")
    
    if hasattr(board, "set_tile_colour"):
        board.set_tile_colour(x, y, colour)
    else:
        if hasattr(board, "move_at"):
            board.move_at(x, y, colour)
        elif hasattr(board, "apply_move"):
            try:
                board.apply_move(x, y, colour)
            except TypeError:
                board.apply_move(move, colour)
        else:
            raise AttributeError("Board lacks set_tile_colour/move_at/apply_move")

    if board.tiles[x][y].colour != colour:
        raise RuntimeError(f"Move application failed: tile {(x,y)} not set to {colour}")

def play_single_game(agent1_class, agent2_class, board_size=11, verbose=False, turn_cap=None):
    board = Board(board_size)

    a1 = agent1_class(Colour.RED) if isinstance(agent1_class, type) else agent1_class
    a2 = agent2_class(Colour.BLUE) if isinstance(agent2_class, type) else agent2_class

    if a1 is a2:
        a2 = agent2_class(Colour.BLUE) if isinstance(agent2_class, type) else agent2_class

    current_agent = a1
    other_agent = a2
    opp_move = None
    turn = 1

    if turn_cap is None:
        turn_cap = board_size * board_size + 10

    while True:
        start = time.time()
        try:
            move = current_agent.make_move(turn, board, opp_move)
        except Exception as e:
            if verbose:
                print(f"[T{turn}] ERROR: agent threw exception in make_move: {e}")
            return "BLUE" if current_agent.colour == Colour.RED else "RED"
        dur = time.time() - start

        if verbose:
            print(f"[T{turn}] {current_agent.__class__.__name__} ({Colour.get_char(current_agent.colour)}) -> {getattr(move,'x',None)},{getattr(move,'y',None)} ({dur:.3f}s)")

        if move is None:
            if verbose:
                print("Agent returned None -> opponent wins")
            return "BLUE" if current_agent.colour == Colour.RED else "RED"

        # handle swap
        if move.x == -1 and move.y == -1:
            if hasattr(board, "swap_sides"):
                board.swap_sides()
                if hasattr(a1, "colour"):
                    a1.colour, a2.colour = a2.colour, a1.colour
            else:
                if verbose:
                    print("Swap requested but board.swap_sides() unavailable -> illegal")
                return "BLUE" if current_agent.colour == Colour.RED else "RED"
        else:
            # Try to apply move and catch any problem
            try:
                _apply_move_board(board, move, current_agent.colour)
            except Exception as e:
                if verbose:
                    print("Error applying move:", e)
                return "BLUE" if current_agent.colour == Colour.RED else "RED"

        try:
            if board.has_ended(Colour.RED):
                return "RED"
            if board.has_ended(Colour.BLUE):
                return "BLUE"
        except Exception:
            if hasattr(board, "get_winner") and board.get_winner() is not None:
                return "RED" if board.get_winner() == Colour.RED else "BLUE"

        # prepare next turn
        opp_move = move
        current_agent, other_agent = other_agent, current_agent
        turn += 1

        if turn > turn_cap:
            # tie-breaker by piece count
            red_count = 0
            blue_count = 0
            if hasattr(board, "tiles"):
                for row in board.tiles:
                    for tile in row:
                        if getattr(tile, "colour", None) == Colour.RED:
                            red_count += 1
                        elif getattr(tile, "colour", None) == Colour.BLUE:
                            blue_count += 1
            if verbose:
                print("Turn cap reached; deciding by piece counts:", red_count, blue_count)
            return "RED" if red_count >= blue_count else "BLUE"
def run_matches(agent1_class, agent2_class, games, board_size=11, verbose=False):
    
    results = {"agent1": 0, "agent2": 0}

    # To handle odd numbers, agent1 will get one extra game as Red
    games_as_p1_red = (games + 1) // 2
    games_as_p2_red = games - games_as_p1_red

    # Phase 1: agent1 = Red, agent2 = Blue
    for i in range(games_as_p1_red):
        if verbose:
            print(f"=== Game {i+1}/{games} (agent1=Red, agent2=Blue) ===")
        winner = play_single_game(agent1_class, agent2_class,
                                  board_size=board_size, verbose=verbose)
        if winner == "RED":
            results["agent1"] += 1
        else:
            results["agent2"] += 1

    # Phase 2: agent2 = Red, agent1 = Blue
    for i in range(games_as_p2_red):
        if verbose:
            print(f"=== Game {games_as_p1_red + i + 1}/{games} "
                  f"(agent2=Red, agent1=Blue) ===")
        winner = play_single_game(agent2_class, agent1_class,
                                  board_size=board_size, verbose=verbose)
        if winner == "RED":
            # Red is agent2 in this phase
            results["agent2"] += 1
        else:
            # Blue is agent1 in this phase
            results["agent1"] += 1

    return results


# Elo helpers and round robin
def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def update_elo(r_a, r_b, score_a, k=32):
    e_a = expected_score(r_a, r_b)
    return (r_a + k * (score_a - e_a), r_b + k * ((1 - score_a) - (1 - e_a)))

def round_robin_elos(agent_classes, games_each, board_size=11, verbose=False):
    elos = {name: 1200.0 for name, _ in agent_classes}
    results_matrix = defaultdict(lambda: defaultdict(int))

    for i in range(len(agent_classes)):
        for j in range(i+1, len(agent_classes)):
            name_a, class_a = agent_classes[i]
            name_b, class_b = agent_classes[j]

            if verbose:
                print(f"Matchup {name_a} (Red) vs {name_b} (Blue)")
            res1 = run_matches(class_a, class_b, games_each, board_size=board_size, verbose=verbose)
            wins_a = res1["agent1"]
            wins_b = res1["agent2"]

            total = wins_a + wins_b
            if total > 0:
                score_a = wins_a / total
                elos[name_a], elos[name_b] = update_elo(elos[name_a], elos[name_b], score_a)

            results_matrix[name_a][name_b] += wins_a
            results_matrix[name_b][name_a] += wins_b

            if verbose:
                print(f"Matchup {name_b} (Red) vs {name_a} (Blue)")
            res2 = run_matches(class_b, class_a, games_each, board_size=board_size, verbose=verbose)
            wins_b2 = res2["agent1"]
            wins_a2 = res2["agent2"]

            total2 = wins_a2 + wins_b2
            if total2 > 0:
                score_a2 = wins_a2 / total2
                elos[name_a], elos[name_b] = update_elo(elos[name_a], elos[name_b], score_a2)

            results_matrix[name_a][name_b] += wins_a2
            results_matrix[name_b][name_a] += wins_b2

    return elos, results_matrix
