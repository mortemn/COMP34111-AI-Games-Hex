from src.Colour import Colour
from src.Move import Move
from random import choice

RED_FIRST_MOVES = [Move(6, 6), Move(5, 6), Move(6, 5), Move(4, 4), Move(4, 5), Move(5, 4)]
BLUE_COUNTERS = {
    (5, 5): [Move(5, 6), Move(6, 5), Move(4, 5), Move(5, 4)],
    (5, 6): [Move(6, 6), Move(5, 5), Move(4, 5)],
    (6, 5): [Move(5, 4), Move(6, 6), Move(5, 5)],
    (4, 5): [Move(6, 5), Move(5, 5), Move(4, 4)],
    (5, 4): [Move(6, 6), Move(5, 5), Move(4, 5)],
    (6, 6): [Move(5, 5), Move(5, 6), Move(6, 5)],
    (4, 4): [Move(5, 5), Move(4, 5), Move(5, 4)],
    (6, 4): [Move(5, 5), Move(6, 5)],
    (4, 6): [Move(5, 5), Move(5, 6), Move(4, 5)],
}

# Very minimal opening book implementation
class OpeningBook():
    def __init__(self, colour: Colour):
        self.colour = colour

    def in_book(self, turn: int, last_move: Move):
        if self.colour == Colour.BLUE and turn == 2 and last_move is not None:
            return True

        if self.colour == Colour.RED and turn == 1:
            return True

        return False

    def should_swap(self, last_move: Move):
        return (last_move.x, last_move.y) in ((5, 5), (6, 5), (5, 6), (4, 5), (5, 4))

    def play_move(self, turn: int, last_move: Move):
        if turn == 1 and self.colour == Colour.RED:
            return choice(RED_FIRST_MOVES)

        if turn == 2 and self.colour == Colour.BLUE and last_move is not None:
            if self.should_swap(last_move):
                return Move(-1, -1)

            if (last_move.x, last_move.y) in BLUE_COUNTERS:
                return choice(BLUE_COUNTERS[(last_move.x, last_move.y)])
            else:
                center = [
                    Move(5, 5), Move(5, 6), Move(6, 5), Move(6, 6),
                    Move(4, 5), Move(5, 4), Move(4, 4)
                ]

                legal_moves = [m for m in center if not (m.x == last_move.x and m.y == last_move.y)]
                
                if legal_moves:
                    return choice(legal_moves)
                else:
                    return Move(5, 5)
        return None
