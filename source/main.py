"""
Entry point.
Run the solver on synthetic graphs and print results.
"""

import time
import networkx as nx

from solver import MVCSolver
from graph_io import generate_synthetic, load_edge_list


def run_single(G: nx.Graph, label: str) -> dict:
    """
    Solve MVC on G, print a summary, return result dict.
    """
    solver = MVCSolver()

    print(f"\n{'='*55}")
    print(f"Graph : {label}")
    print(f"Nodes : {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")

    start = time.perf_counter()
    cover, nodes_visited = solver.solve(G)
    elapsed = time.perf_counter() - start

    # Verify correctness
    valid = all(
        u in cover or v in cover
        for u, v in G.edges()
    )

    print(f"Cover size    : {len(cover)}")
    print(f"Tree nodes    : {nodes_visited:,}")
    print(f"Time (s)      : {elapsed:.4f}")
    print(f"Valid cover   : {valid}")

    return {
        "label": label,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "cover_size": len(cover),
        "tree_nodes_visited": nodes_visited,
        "time_seconds": elapsed,
        "valid": valid,
    }


def main():
    results = []

    # ----------------------------------------------------------------
    # Synthetic test graphs
    # ----------------------------------------------------------------
    test_cases = [
        ("path_20",
         generate_synthetic("path", n=20)),

        ("cycle_20",
         generate_synthetic("cycle", n=20)),

        ("erdos_renyi_50_005",
         generate_synthetic("erdos_renyi", n=50, p=0.05)),

        ("erdos_renyi_80_003",
         generate_synthetic("erdos_renyi", n=80, p=0.03)),

        ("barabasi_albert_100_2",
         generate_synthetic("barabasi_albert", n=100, m=2)),

        ("random_regular_3_60",
         generate_synthetic("random_regular", d=3, n=60)),
    ]

    for label, G in test_cases:
        result = run_single(G, label)
        results.append(result)

    # ----------------------------------------------------------------
    # Optional: load a real graph file
    # Uncomment and adjust path as needed:
    # ----------------------------------------------------------------
    # G_real = load_edge_list("data/email-Enron.txt.gz")
    # results.append(run_single(G_real, "email-Enron"))

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"{'Label':<30} {'|V|':>6} {'|E|':>7} "
          f"{'Cover':>6} {'Nodes':>8} {'Time':>8}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['label']:<30} "
            f"{r['graph_nodes']:>6} "
            f"{r['graph_edges']:>7} "
            f"{r['cover_size']:>6} "
            f"{r['tree_nodes_visited']:>8,} "
            f"{r['time_seconds']:>7.4f}s"
        )


if __name__ == "__main__":
    main()