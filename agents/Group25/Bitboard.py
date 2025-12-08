from src.Board import Board
from src.Colour import Colour

NEIGHBOURS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

def precompute(size):
    neighbour_table = {}

    for x in range(size):
        for y in range(size):
            idx = x * size + y
            mask = 0
            for dx, dy in NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    nidx = nx * size + ny
                    mask |= 1 << nidx
            neighbour_table[idx] = mask
    return neighbour_table

MASKS = precompute(11)

def row_mask(size, x):
    return sum(1 << (x * size + y) for y in range(size))

def col_mask(size, y):
    return sum(1 << (x * size + y) for x in range(size))

TOP_MASK = row_mask(11, 0)
BOTTOM_MASK = row_mask(11, 10)
LEFT_MASK = col_mask(11, 0)
RIGHT_MASK = col_mask(11, 10)

class Bitboard():
    def __init__(self, size: int = 11, red: int = 0, blue: int = 0):
        self.size = size
        self.red = red
        self.blue = blue
        self.empty_cells = set((x, y) for x in range(size) for y in range(size))
        self.red_moves = 0
        self.blue_moves = 0

    def index(self, x: int, y: int):
        return x * self.size + y

    def coord(self, i: int):
        x = i // self.size
        y = i % self.size
        return (x, y)

    def bit(self, x, y):
        return 1 << self.index(x, y)

    def colour_at(self, x: int, y: int):
        b = self.bit(x, y)
        if self.red & b:
            return Colour.RED
        elif self.blue & b:
            return Colour.BLUE 
        else:
            return None

    def occupied(self):
        return self.red | self.blue

    def move_at(self, x: int, y: int, colour: Colour):
        b = self.bit(x, y)
        if colour == Colour.RED:
            self.red |= b
            self.red_moves += 1
        elif colour == Colour.BLUE:
            self.blue |= b
            self.blue_moves += 1
        self.empty_cells.discard((x, y))

    def undo_at(self, x: int, y: int, colour: Colour):
        b = self.bit(x, y)
        if colour == Colour.RED:
            self.red &= ~b
            self.red_moves -= 1
        elif colour == Colour.BLUE:
            self.blue &= ~b
            self.blue_moves -= 1
        self.empty_cells.add((x, y))

    def red_can_win(self):
        return self.red_moves >= self.size - 1

    def blue_can_win(self):
        return self.blue_moves >= self.size - 1

    def legal_moves(self):
        return list(self.empty_cells)

    def copy(self):
        new_board = Bitboard(self.size, self.red, self.blue)
        new_board.empty_cells = self.empty_cells.copy()
        return new_board

    def red_won(self):
        red = self.red
        if red == 0: return False

        frontier = red & TOP_MASK
        visited = 0

        while frontier:
            if frontier & BOTTOM_MASK:
                return True
            visited |= frontier

            new_frontier = 0
            while frontier:
                lsb = frontier & -frontier
                frontier ^= lsb
                idx = (lsb).bit_length() - 1

                neighbours = MASKS[idx] & red & ~visited
                new_frontier |= neighbours
                visited |= neighbours

            frontier = new_frontier

        return False

    def blue_won(self):
        blue = self.blue
        if blue == 0: return False

        frontier = blue & LEFT_MASK
        visited = 0

        while frontier:
            if frontier & RIGHT_MASK:
                return True
            visited |= frontier

            new_frontier = 0
            while frontier:
                lsb = frontier & -frontier
                frontier ^= lsb
                idx = (lsb).bit_length() - 1

                neighbours = MASKS[idx] & blue & ~visited
                new_frontier |= neighbours
                visited |= neighbours

            frontier = new_frontier

        return False

def convert_bitboard(board: Board):
    size = board.size
    red = 0
    blue = 0
    empty_cells = set()

    for x in range(size):
        for y in range(size):
            tile = board.tiles[x][y]
            idx = x * size + y

            if tile.colour == Colour.RED:
                red |= 1 << idx
            elif tile.colour == Colour.BLUE:
                blue |= 1 << idx
            else:
                empty_cells.add((x, y))

    bitboard = Bitboard(size, red, blue)
    bitboard.empty_cells = empty_cells
    return bitboard
