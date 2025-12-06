import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Benchmarksuite import run_matches, round_robin_elos
from RAVEMCTSAgent import MCTSAgent
from ReuseAndRAVE import MCTSAgent

if __name__ == "__main__":
    print("Starting quick benchmark.")
    agent1 = MCTSAgent
    agent2 = MCTSAgent

    results = run_matches(
        agent1_class=agent1,
        agent2_class=agent2,
        games=3,
        board_size=11
    )

    print("Match Results:")
    print(results)
    
    print("\n=== Elo Round Robin Tournament ===")

    agents = [
        ("RAVEMCTSAgent", MCTSAgent),
        ("ReuseAndRAVEMCTSAgent", MCTSAgent),
    ]

    elos, matrix = round_robin_elos(
        agent_classes=agents,
        games_each=2
    )

    print("\nFinal Elo Ratings:")
    for name, elo in elos.items():
        print(f"  {name}: {elo:.1f}")

    print("\nHead-to-Head Win Matrix:")
    for a in matrix:
        print(f"{a}: {dict(matrix[a])}")
