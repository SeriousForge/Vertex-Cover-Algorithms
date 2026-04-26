"""
Entry point.
Run the solver on synthetic graphs and print results.
"""

import time
import networkx as nx

from solver import MVCSolverNaive, MVCSolverComponentAware
from graph_io import generate_synthetic, load_edge_list

def run_comparison(G: nx.graph, label: str)-> dict:
     
    print(f"\n{'='*62}")
    print(f"Graph : {label}")
    print(f"Nodes : {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
    print(f"{'':-<62}")

    solver_naive = MVCSolverNaive()
    t0 = time.perf_counter()
    cover_n, nodes_n = solver_naive.solve(G)
    time_n = time.perf_counter() - t0
    valid_n = all(u in cover_n or v in cover_n for u, v in G.edges())

    solver_aware = MVCSolverComponentAware()
    t0 = time.perf_counter()
    cover_a, nodes_a = solver_aware.solve(G)
    time_a = time.perf_counter() - t0
    valid_a = all(u in cover_a or v in cover_a for u, v in G.edges())

    node_reduction = 1 - nodes_a / max(nodes_n, 1)
    time_speedup = time_n / max(time_a, 1e-9)

    print(f"{'Metric':<28} {'Naive':>10}  {'Comp-Aware':>10}")
    print(f"{'':28} {'':>10}  {'':>10}")
    print(f"{'Cover size':<28} {len(cover_n):>10}  {len(cover_a):>10}")
    print(f"{'Tree nodes visited':<28} {nodes_n:>10,}  {nodes_a:>10,}")
    print(f"{'Time (s)':<28} {time_n:>10.4f}  {time_a:>10.4f}")
    print(f"{'Valid cover':<28} {str(valid_n):>10}  {str(valid_a):>10}")
    print(f"{'':28} {'':>10}  {'':>10}")
    print(f"{'Node count reduction':<28} {'':>10}  {node_reduction:>9.1%}")
    print(f"{'Speedup (naive/aware)':<28} {'':>10}  {time_speedup:>9.2f}x")

    return {
        "label":          label,
        "graph_nodes":    G.number_of_nodes(),
        "graph_edges":    G.number_of_edges(),
        "cover_size":     len(cover_a),
        "naive_nodes":    nodes_n,
        "aware_nodes":    nodes_a,
        "naive_time":     time_n,
        "aware_time":     time_a,
        "node_reduction": node_reduction,
        "speedup":        time_speedup,
        "valid":          valid_n and valid_a,
    }

def run_single(G: nx.Graph, label: str, use_aware: bool = True) -> dict:
    """
    Solve MVC on G, print a summary, return result dict.
    """
    solver = MVCSolverComponentAware() if use_aware else MVCSolverNaive()
    solver_name = "Component-Aware" if use_aware else "Naive"

    print(f"\n{'='*55}")
    print(f"Graph  : {label}  [{solver_name}]")
    print(f"Nodes : {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")

    t0 = time.perf_counter()
    cover, nodes_visited = solver.solve(G)
    elapsed = time.perf_counter() - t0

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
        result = run_comparison(G, label)
        results.append(result)


    # G_real = load_edge_list("data/email-Enron.txt.gz")
    # results.append(run_single(G_real, "email-Enron"))

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print(f"\n\n{'='*90}")
    print("SUMMARY — Naive vs Component-Aware")
    print(f"{'='*90}")
    print(
        f"{'Label':<28} {'|V|':>5} {'|E|':>6} "
        f"{'Cover':>6} "
        f"{'Naive Nodes':>12} {'Aware Nodes':>12} "
        f"{'Reduction':>10} "
        f"{'Naive(s)':>9} {'Aware(s)':>9} {'Speedup':>8}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['label']:<28} "
            f"{r['graph_nodes']:>5} "
            f"{r['graph_edges']:>6} "
            f"{r['cover_size']:>6} "
            f"{r['naive_nodes']:>12,} "
            f"{r['aware_nodes']:>12,} "
            f"{r['node_reduction']:>9.1%} "
            f"{r['naive_time']:>9.4f} "
            f"{r['aware_time']:>9.4f} "
            f"{r['speedup']:>7.2f}x"
        )
    print(f"{'='*90}")

if __name__ == "__main__":
    main()