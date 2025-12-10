
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Benchmarksuite import run_matches, round_robin_elos
from EliminateCopy import EliminateCopyMCTS
from BaseMCTSAgent import BaseMCTSAgent
from RAVEMCTSAgent import MCTSAgent
from ReuseAndRAVE import ReuseMCTS
from IntegrateBridge import IntegratedAgent
from EndgameMCTS import EndgameMCTSAgent as EndgameMCTSAgent
from HeuMCTS import HeuMCTSAgent
from TunedHeuristics import TunedMCTSAgent as TunedHeuristicMCTSAgent
from OrderedMCTS import OrderedMCTS
from SmallHeuristics import SmallHeuMCTS
from ImprovedHeuristics import ImprovedHeuMCTS

if __name__ == "__main__":
    print("Starting quick benchmark.")
    agent1 = MCTSAgent
    agent2 = EliminateCopyMCTS

    results = run_matches(
        agent1_class=agent1,
        agent2_class=agent2,
        games=150,
        board_size=11,
        verbose=True
    )

    print("\nMatch Results:")
    print(results)

    # agents = [ ("Endgame", EndgameMCTSAgent), ("RAVE", RAVEMCTSAgent), ("RootParallel", RootParallelMCTSAgent), ]
    # print("\nRunning round-robin Elo tournament (2 games per pairing)...") 
    # elos, matrix = round_robin_elos(agents, games_each=2, board_size=11, verbose=False) 
    # print("\nElo Ratings:") 
    # for k, v in elos.items(): 
    #     print(f"{k}: {v:.1f}") 
    # print("\nWin matrix:") 
    # for k in matrix: 
    #     print(k, dict(matrix[k]))
