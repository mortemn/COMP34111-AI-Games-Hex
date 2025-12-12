import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Colour import Colour
from src.Board import Board
from src.Move import Move
from agents.Group25.EliminateCopy import EliminateCopyMCTS

if __name__ == "__main__":
    # Agent colour
    colour = Colour.BLUE
    agent = EliminateCopyMCTS(colour)
    turn = 0
    board = Board(11)

    while True:
        user_input = input("Enter your move:")
        x, y = map(int, user_input.split(","))
        board.set_tile_colour(x, y, Colour.opposite(colour))
        move = agent.make_move(turn, board, Move(x, y))
        board.set_tile_colour(move.x, move.y, colour)
        turn += 1
