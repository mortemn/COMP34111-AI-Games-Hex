from src.Colour import Colour
from src.Move import Move

# Very minimal opening book implementation
class OpeningBook():
    def __init__(self, colour: Colour):
        self.colour = colour

    def in_book(self, turn: int, last_move: Move):
        if self.colour == Colour.BLUE:
            if turn == 2 and self.dist_from_center(last_move) <= 1:
                return True
        if self.colour == Colour.RED:
            if turn == 1:
                return True
        return False

    def dist_from_center(self, move: Move):
        return max(abs(move.x - 5), abs(move.y - 5))

    def play_move(self, turn: int, last_move: Move):
        if turn == 1 and self.colour == Colour.RED:
            return Move(5, 6)
        if turn == 2 and self.colour == Colour.BLUE:
            if self.dist_from_center(last_move) <= 1:
                return Move(-1, -1)
