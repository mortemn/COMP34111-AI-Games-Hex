from src.Colour import Colour
from src.Move import Move
from random import choice

RED_FIRST_MOVES = [Move(6, 6), Move(5, 6), Move(6, 5), Move(4, 4), Move(4, 5), Move(5, 4)]
BLUE_CLAIM_MOVES = [Move(5, 5), Move(5, 6), Move(6, 5), Move(6, 6)]
BLUE_COUNTERS = {
    Move(5, 7): Move(6, 5),
    Move(5, 3): Move(4, 5),
    Move(7, 5): Move(6, 6),
    Move(3, 5): Move(5, 4),
    Move(4, 6): Move(6, 5),
    Move(6, 6): Move(4, 5),
    Move(6, 4): Move(5, 6),
    Move(4, 4): Move(6, 5)
}

# Very minimal opening book implementation
class OpeningBook():
    def __init__(self, colour: Colour):
        self.colour = colour

    def in_book(self, turn: int, last_move: Move):
        if self.colour == Colour.BLUE and turn == 2:
            d = self.dist_from_center(last_move)
            if d <= 2:
                return True
            # If red plays away from the center, stake a claim there
            if d >= 3:
                return True
        if self.colour == Colour.RED:
            if turn == 1:
                return True
        return False

    def dist_from_center(self, move: Move):
        return max(abs(move.x - 5), abs(move.y - 5))

    def play_move(self, turn: int, last_move: Move):
        if turn == 1 and self.colour == Colour.RED:
            return choice(RED_FIRST_MOVES)
        if turn == 2 and self.colour == Colour.BLUE:
            d = self.dist_from_center(last_move)
            if d <= 1:
                return Move(-1, -1)

            if d == 2:
                move = BLUE_COUNTERS.get(last_move, choice(BLUE_CLAIM_MOVES))
                if move:
                    return move

            if d >= 3:
                return choice(BLUE_CLAIM_MOVES)
        return Move(5, 5)
