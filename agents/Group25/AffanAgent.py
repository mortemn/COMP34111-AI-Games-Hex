from __future__ import annotations

from math import sqrt, inf, log
from random import choice
from time import perf_counter
from heapq import heappush, heappop

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Game import Game
from src.Move import Move

from agents.Group25.affan_board import AffanBoardState


class Node:
    # Only use RAVE near the root – deeper nodes use plain UCT
    RAVE_MAX_DEPTH = 4
    # Only use the first N moves from a playout for RAVE updates
    RAVE_MAX_PLAYOUT_MOVES = 20

    # Control how rollouts are cut off / evaluated
    MAX_ROLLOUT_STEPS = 40          # how many random moves before we stop
    HEURISTIC_EMPTY_CUTOFF = 8      # when board is almost full, use strong heuristic

    def __init__(
        self,
        colour: Colour,
        move: Move | None,
        parent: "Node | None",
        state: AffanBoardState,
        root_colour: Colour,
        depth: int = 0,
    ):
        # Colour to move in the current turn
        self.colour = colour
        # Last move that was made (None for root)
        self.move = move
        # Parent of the current node
        self.parent = parent

        # Internal lightweight board
        self.state: AffanBoardState = state
        self.grid = state.grid
        self.size_x = state.size
        self.size_y = state.size

        # Tree depth (root = 0)
        self.depth = depth

        # MCTS stats
        self.visits = 0
        self.wins = 0
        self.children: list[Node] = []
        self.root_colour = root_colour
        self.c = sqrt(2)

        # --- RAVE / AMAF stats as fixed-size arrays (only for shallow nodes) ---
        self.board_size = state.size
        if self.depth < Node.RAVE_MAX_DEPTH:
            n_cells = self.board_size * self.board_size
            self.rave_wins = [0.0] * n_cells
            self.rave_visits = [0.0] * n_cells
        else:
            # No RAVE for deeper nodes
            self.rave_wins = None
            self.rave_visits = None

        # Untried moves from current position, ordered by local bridge heuristic
        moves = [Move(x, y) for (x, y) in self.state.get_empty_cells()]
        if moves:
            moves.sort(
                key=lambda m: -self._bridge_score(
                    self.grid,
                    self.size_x,
                    self.size_y,
                    m,
                    self.colour,
                )
            )
        self.untried_moves: list[Move] = moves

    # ---------- helpers ----------

    def _move_index(self, move: Move) -> int:
        """Map (x, y) to a flat index for RAVE arrays."""
        return move.x * self.board_size + move.y

    def _colour_to_int(self, colour: Colour | None) -> int:
        if colour == Colour.RED:
            return AffanBoardState.RED
        if colour == Colour.BLUE:
            return AffanBoardState.BLUE
        return AffanBoardState.EMPTY  # empty

    # ---------- neighbours on the grid ----------

    def _grid_neighbours(self, size_x: int, size_y: int, x: int, y: int):
        """
        Neighbour coordinates on the integer grid (Hex adjacency).

        We ignore size_x/size_y here and use the precomputed neighbour
        lists in AffanBoardState for speed.
        """
        for nx, ny in self.state.neighbours[x][y]:
            yield nx, ny

    # --------- bridge heuristic on the grid ---------

    def _bridge_score(
        self,
        grid,
        size_x: int,
        size_y: int,
        move: Move,
        colour: Colour,
    ) -> float:
        x, y = move.x, move.y
        own_val = self._colour_to_int(colour)
        opp_val = self._colour_to_int(Colour.opposite(colour))

        own_neighbours = 0
        opp_neighbours = 0
        empty_neighbours = []

        for nx, ny in self._grid_neighbours(size_x, size_y, x, y):
            v = grid[nx][ny]
            if v == own_val:
                own_neighbours += 1
            elif v == opp_val:
                opp_neighbours += 1
            else:
                empty_neighbours.append((nx, ny))

        # extend own groups (3) + block opponent (2)
        score = 3 * own_neighbours + 2 * opp_neighbours

        # very simple "bridge" pattern bonus
        if own_neighbours > 0 and len(empty_neighbours) >= 2:
            own_neighs_sets = []
            for ex, ey in empty_neighbours:
                s = set()
                for nx, ny in self._grid_neighbours(size_x, size_y, ex, ey):
                    if grid[nx][ny] == own_val:
                        s.add((nx, ny))
                own_neighs_sets.append(s)

            for i in range(len(own_neighs_sets)):
                for j in range(i + 1, len(own_neighs_sets)):
                    if own_neighs_sets[i] & own_neighs_sets[j]:
                        score += 1.0

        return score

    def _choose_bridge_move(self, grid, size_x: int, size_y: int, moves, colour: Colour) -> Move:
        from math import inf  # ensure inf is in scope here if not imported above
        best_score = -inf
        best_moves = []

        for m in moves:
            s = self._bridge_score(grid, size_x, size_y, m, colour)
            if s > best_score:
                best_score = s
                best_moves = [m]
            elif s == best_score:
                best_moves.append(m)

        if not best_moves:
            return choice(moves)
        return choice(best_moves)

    # --------- static evaluation (shortest path, expensive but strong) ----------

    def _shortest_path_cost(self, grid, size_x: int, size_y: int, colour: Colour) -> float:
        own_val = self._colour_to_int(colour)
        opp_val = self._colour_to_int(Colour.opposite(colour))

        INF = 1e9
        dist = [[INF] * size_y for _ in range(size_x)]
        heap = []

        if colour == Colour.RED:
            # RED: top row → bottom row
            for y in range(size_y):
                if grid[0][y] == opp_val:
                    continue
                cost = 0 if grid[0][y] == own_val else 1
                dist[0][y] = cost
                heappush(heap, (cost, 0, y))

            def goal_check(x, y): return x == size_x - 1

        else:  # BLUE: left col → right col
            for x in range(size_x):
                if grid[x][0] == opp_val:
                    continue
                cost = 0 if grid[x][0] == own_val else 1
                dist[x][0] = cost
                heappush(heap, (cost, x, 0))

            def goal_check(x, y): return y == size_y - 1

        while heap:
            cost, x, y = heappop(heap)
            if cost > dist[x][y]:
                continue
            if goal_check(x, y):
                return cost

            for nx, ny in self._grid_neighbours(size_x, size_y, x, y):
                if grid[nx][ny] == opp_val:
                    step = 50  # big penalty for going through opponent
                elif grid[nx][ny] == own_val:
                    step = 0
                else:
                    step = 1
                new_cost = cost + step
                if new_cost < dist[nx][ny]:
                    dist[nx][ny] = new_cost
                    heappush(heap, (new_cost, nx, ny))

        return INF

    def _heuristic_winner(self, grid, size_x: int, size_y: int) -> Colour:
        """
        Strong (but expensive) static evaluation:
        compare shortest connection costs for root vs opponent.
        Used only when the board is almost full.
        """
        root = self.root_colour
        d_root = self._shortest_path_cost(grid, size_x, size_y, root)
        d_opp = self._shortest_path_cost(
            grid, size_x, size_y, Colour.opposite(root)
        )

        if d_root < d_opp:
            return root
        elif d_opp < d_root:
            return Colour.opposite(root)
        else:
            return root  # tie-break towards root

    def _fast_score_winner(self, grid, size: int) -> Colour:
        """
        Cheap fallback evaluation:
        just counts stones on the board (biased to root),
        no heaps, no matrices.
        """
        root = self.root_colour
        opp = Colour.opposite(root)

        root_val = self._colour_to_int(root)
        opp_val = self._colour_to_int(opp)

        root_score = 0
        opp_score = 0

        for x in range(size):
            row = grid[x]
            for y in range(size):
                v = row[y]
                if v == root_val:
                    root_score += 2
                elif v == opp_val:
                    opp_score += 2

        if root_score >= opp_score:
            return root
        else:
            return opp

    # --------- MCTS core with (optimised) RAVE ----------

    def ucb(self, child: "Node") -> float:
        """
        UCB with RAVE / AMAF mixing at shallow depths.

        child.wins / child.visits = direct stats.
        self.rave_*[idx(move)]    = AMAF stats at this node.
        """
        # Deep in the tree or no move: plain UCT
        if self.depth >= Node.RAVE_MAX_DEPTH or child.move is None:
            if child.visits == 0:
                return inf
            q = child.wins / child.visits
            return q + self.c * sqrt(log(self.visits + 1) / child.visits)

        # RAVE-enhanced UCT near the top
        idx = self._move_index(child.move)
        rave_n = self.rave_visits[idx]
        rave_q = (self.rave_wins[idx] / rave_n) if rave_n > 0 else 0.5

        if child.visits == 0:
            # Pure RAVE for completely unvisited children if we have info,
            # otherwise force exploration.
            if rave_n > 0:
                return rave_q + self.c * sqrt(log(self.visits + 1))
            return inf

        q = child.wins / child.visits

        # Standard β schedule: more weight on RAVE early on
        k = 300.0
        beta = 0.0
        if rave_n > 0:
            beta = rave_n / (child.visits + rave_n +
                             4.0 * child.visits * rave_n / k)

        mixed_q = (1.0 - beta) * q + beta * rave_q
        return mixed_q + self.c * sqrt(log(self.visits + 1) / child.visits)

    def best_child(self):
        return max(self.children, key=lambda x: self.ucb(x))

    def select(self):
        node = self
        while not node.untried_moves and node.children:
            node = node.best_child()
        return node

    def expand(self):
        move = self.untried_moves.pop(0)

        new_state = self.state.copy()
        new_state.play(move.x, move.y, self.colour)

        new_node = Node(
            colour=Colour.opposite(self.colour),
            move=move,
            parent=self,
            state=new_state,
            root_colour=self.root_colour,
            depth=self.depth + 1,
        )
        self.children.append(new_node)
        return new_node

    def simulate(self) -> tuple[Colour, list[tuple[Colour, int]]]:
        """
        Playout starting from this node's state.

        Returns:
            (winner, played_moves)
            where played_moves is a list of (colour, cell_index) for AMAF/RAVE.
            cell_index = x * size + y
        """
        state = self.state.copy()
        size = state.size
        colour = self.colour

        steps = 0
        played_moves: list[tuple[Colour, int]] = []

        while True:
            winner = state.get_winner()
            if winner is not None:
                return winner, played_moves

            empties = state.get_empty_cells()
            if not empties:
                return Colour.opposite(colour), played_moves

            # 1) If the board is almost full, use the strong (Dijkstra) heuristic.
            if len(empties) <= Node.HEURISTIC_EMPTY_CUTOFF:
                heuristic_winner = self._heuristic_winner(
                    state.grid, size, size
                )
                return heuristic_winner, played_moves

            # 2) If rollout is getting long, use a cheap score heuristic.
            if steps >= Node.MAX_ROLLOUT_STEPS:
                cheap_winner = self._fast_score_winner(state.grid, size)
                return cheap_winner, played_moves

            # --- FAST MOVE SELECTION FOR ROLLOUTS ---
            # Just pick a random empty cell (no bridge heuristic here).
            x, y = choice(empties)

            state.play(x, y, colour)
            cell_index = x * size + y
            played_moves.append((colour, cell_index))

            colour = Colour.opposite(colour)
            steps += 1

    def backpropagate(self, winner: Colour, played_moves: list[tuple[Colour, int]]):
        """
        Standard backprop for UCT + RAVE backprop on all moves
        from the playout that belong to the player-to-move at this node.

        played_moves contains (colour, cell_index).
        """
        self.visits += 1
        if winner == self.root_colour:
            self.wins += 1

        # --- RAVE / AMAF update only for shallow nodes ---
        if self.depth < Node.RAVE_MAX_DEPTH and self.rave_visits is not None:
            player = self.colour
            seen: set[int] = set()

            for i, (move_colour, cell_index) in enumerate(played_moves):
                if i >= Node.RAVE_MAX_PLAYOUT_MOVES:
                    break
                if move_colour != player:
                    continue

                if cell_index in seen:
                    continue
                seen.add(cell_index)

                self.rave_visits[cell_index] += 1.0
                if winner == player:
                    self.rave_wins[cell_index] += 1.0

        if self.parent:
            self.parent.backpropagate(winner, played_moves)

    def search(self, time_limit: float) -> tuple["Node", int]:
        """
        Run MCTS (UCT + RAVE) until the given time budget is exhausted.

        Returns:
            (best_child, iterations)
        """
        start = perf_counter()
        i = 0

        while True:
            if i > 0 and perf_counter() - start >= time_limit:
                break

            node = self.select()
            if node.untried_moves:
                node = node.expand()
            winner, played_moves = node.simulate()
            node.backpropagate(winner, played_moves)

            i += 1

        # If something weird happens and we have no children, just play any move
        if not self.children:
            empties = self.state.get_empty_cells()
            x, y = choice(empties)
            dummy_state = self.state.copy()
            dummy_state.play(x, y, self.colour)
            dummy_node = Node(
                colour=Colour.opposite(self.colour),
                move=Move(x, y),
                parent=self,
                state=dummy_state,
                root_colour=self.root_colour,
                depth=self.depth + 1,
            )
            return dummy_node, i

        best = max(self.children, key=lambda x: x.visits)
        return best, i


class BetterAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.root_node: Node | None = None
        self.time_used: float = 0.0
        # Persistent internal state mirroring the real game
        self.shadow_state: AffanBoardState | None = None

    def _compute_time_limit(self, turn: int, board: Board) -> float:
        """
        Phase-based time management over the whole game.

        Safe total budget ≈ 280s (under engine's 300s).
        Early   (turn <= size)   → ~12%, cap ~20s
        Mid     (turn <= 2*size) → ~10%, cap ~12s
        Late    (else)           → ~8%,  cap ~8s
        Panic   (remaining < 5s) → 0.1s
        """
        TOTAL_ENGINE = Game.MAXIMUM_TIME / 1e9
        SAFE_BUDGET = min(280.0, TOTAL_ENGINE)

        remaining_total = max(0.01, SAFE_BUDGET - self.time_used)
        if remaining_total <= 5.0:
            return min(0.1, remaining_total)

        size = board.size
        moves_played = turn

        if moves_played <= size:          # early game
            frac = 0.12
            cap = 20.0
        elif moves_played <= 2 * size:    # mid game
            frac = 0.10
            cap = 12.0
        else:                             # late game
            frac = 0.08
            cap = 8.0

        time_limit = min(cap, remaining_total * frac)
        time_limit = max(0.05, min(time_limit, remaining_total))
        return time_limit

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        size = board.size
        move_start = perf_counter()
        time_limit = self._compute_time_limit(turn, board)

        # --- Initialise / reset shadow state at the start of a game ---
        initialized_now = False
        if self.shadow_state is None or self.shadow_state.size != size or turn == 1:
            # Build from engine board once per game (or on size change)
            self.shadow_state = AffanBoardState.from_engine_board(board)
            self.root_node = None
            initialized_now = True

        # --- Apply opponent move to shadow state (if this isn't the very first init) ---
        if (
            opp_move is not None
            and not opp_move.is_swap()
            and not initialized_now
        ):
            self.shadow_state.play(
                opp_move.x,
                opp_move.y,
                Colour.opposite(self.colour),
            )

        # Smart Red opening (avoid centre to make pie rule less trivial)
        if turn == 1 and opp_move is None and self.colour == Colour.RED:
            c = size // 2
            y = c - 1 if c - 1 >= 0 else c
            move = Move(c, y)
            # update shadow state
            self.shadow_state.play(move.x, move.y, self.colour)
            move_time = perf_counter() - move_start
            self.time_used += move_time
            print(
                f"[BetterAgent] Rollouts this move: 0 (opening), time: {move_time:.4f}s")
            return move

        # Pie rule – turn 2 decision
        if (
            turn == 2
            and opp_move is not None
            and not opp_move.is_swap()
        ):
            total_time = time_limit
            eval_time = total_time * 0.4

            # Use shadow_state instead of rebuilding from engine board
            base_state = self.shadow_state.copy()

            # Non-swap tree
            root_non_swap = Node(
                colour=self.colour,
                move=opp_move,
                parent=None,
                state=base_state,
                root_colour=self.colour,
                depth=0,
            )
            _, it_non_swap = root_non_swap.search(eval_time)
            value_non_swap = (
                root_non_swap.wins / root_non_swap.visits
                if root_non_swap.visits > 0
                else 0.5
            )

            # Swap tree – after swap, our colour flips
            our_after_swap = Colour.opposite(self.colour)
            opp_after_swap = self.colour

            root_swap = Node(
                colour=opp_after_swap,  # opponent to move after we swap
                move=opp_move,
                parent=None,
                state=base_state.copy(),
                root_colour=our_after_swap,
                depth=0,
            )
            _, it_swap = root_swap.search(eval_time)
            value_swap = (
                root_swap.wins / root_swap.visits
                if root_swap.visits > 0
                else 0.5
            )

            print(
                f"[BetterAgent] Turn 2 eval rollouts -> "
                f"non-swap: {it_non_swap}, swap: {it_swap}"
            )

            if value_swap > value_non_swap + 0.02:
                # We choose to swap: no board change, but engine will flip colours.
                self.root_node = None
                move = Move(-1, -1)  # swap
                move_time = perf_counter() - move_start
                self.time_used += move_time
                print(
                    f"[BetterAgent] Rollouts this move: 0 (swap decision), time: {move_time:.4f}s")
                return move

            # No swap: keep non-swap tree and continue searching from it
            self.root_node = root_non_swap
            remaining = max(0.0, total_time - eval_time)
            best_child, iters = self.root_node.search(remaining)
            best_child.parent = None
            self.root_node = best_child
            move = best_child.move
            if not move.is_swap():
                self.shadow_state.play(move.x, move.y, self.colour)
            move_time = perf_counter() - move_start
            self.time_used += move_time
            print(
                f"[BetterAgent] Rollouts this move: {iters}, time: {move_time:.4f}s")
            return move

        # --- Generic tree reuse ---

        if turn == 1:
            self.root_node = None
        if opp_move is not None and opp_move.is_swap():
            # Opponent swapped: board unchanged, but colours flipped by engine.
            self.root_node = None

        if self.root_node is not None and opp_move is not None and not opp_move.is_swap():
            matching_child = None
            for child in self.root_node.children:
                if (
                    child.move is not None
                    and child.move.x == opp_move.x
                    and child.move.y == opp_move.y
                ):
                    matching_child = child
                    break

            if matching_child is not None:
                matching_child.parent = None
                self.root_node = matching_child
            else:
                self.root_node = None

        # Create root node from shadow state if needed
        if self.root_node is None:
            state = self.shadow_state.copy()
            self.root_node = Node(
                colour=self.colour,
                move=opp_move,
                parent=None,
                state=state,
                root_colour=self.colour,
                depth=0,
            )

        best_child, iters = self.root_node.search(time_limit)
        best_child.parent = None
        self.root_node = best_child

        move = best_child.move
        # Apply our chosen move to shadow state
        if not move.is_swap():
            self.shadow_state.play(move.x, move.y, self.colour)

        move_time = perf_counter() - move_start
        self.time_used += move_time
        print(
            f"[BetterAgent] Rollouts this move: {iters}, time: {move_time:.4f}s")
        return move
