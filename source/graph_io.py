"""
Loaders for common graph formats.
"""

import networkx as nx
import gzip
import os


def load_edge_list(filepath: str, comment: str = "#") -> nx.Graph:
    """
    Load a graph from a plain edge list file.
    Each line: 'u v' (whitespace separated).
    Lines starting with `comment` are skipped.

    Works with SNAP .txt and .txt.gz files.
    """
    G = nx.Graph()

    opener = gzip.open if filepath.endswith(".gz") else open

    with opener(filepath, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            parts = line.split()
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))

    return G


def load_dimacs_col(filepath: str) -> nx.Graph:
    """
    Load a DIMACS .col format graph.
    Relevant lines:
      p edge <num_nodes> <num_edges>
      e <u> <v>
    """
    G = nx.Graph()

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("e "):
                parts = line.split()
                G.add_edge(int(parts[1]), int(parts[2]))

    return G


def generate_synthetic(graph_type: str, **kwargs) -> nx.Graph:
    """
    Generate a synthetic graph for testing.

    graph_type options:
        'erdos_renyi'     : kwargs: n, p
        'barabasi_albert' : kwargs: n, m
        'random_regular'  : kwargs: d, n   (d-regular)
        'path'            : kwargs: n
        'cycle'           : kwargs: n
    """
    generators = {
        "erdos_renyi":     lambda: nx.erdos_renyi_graph(
                               kwargs["n"], kwargs["p"]
                           ),
        "barabasi_albert": lambda: nx.barabasi_albert_graph(
                               kwargs["n"], kwargs["m"]
                           ),
        "random_regular":  lambda: nx.random_regular_graph(
                               kwargs["d"], kwargs["n"]
                           ),
        "path":            lambda: nx.path_graph(kwargs["n"]),
        "cycle":           lambda: nx.cycle_graph(kwargs["n"]),
    }

    if graph_type not in generators:
        raise ValueError(
            f"Unknown graph type '{graph_type}'. "
            f"Choose from: {list(generators.keys())}"
        )

    return generators[graph_type]()