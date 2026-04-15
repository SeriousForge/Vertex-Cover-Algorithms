"""
Reduction rules applied before/during branching.
Each rule modifies the graph in-place and returns
the number of vertices forced into the cover.
"""

import networkx as nx


def apply_degree_one_rule(G: nx.Graph, cover: set) -> int:
    """
    Degree-1 rule:
    If a vertex v has degree 1, its single neighbor u
    must be in the cover (u covers more edges).
    Remove both v and u, add u to cover.
    Returns count of vertices added to cover.
    """
    added = 0
    changed = True

    while changed:
        changed = False
        degree_one = [v for v, d in G.degree() if d == 1]

        for v in degree_one:
            if v not in G:          # may have been removed already
                continue
            neighbors = list(G.neighbors(v))
            if len(neighbors) == 0:
                G.remove_node(v)
                continue

            u = neighbors[0]        # the single neighbor
            cover.add(u)
            added += 1

            # remove both v and u from working graph
            G.remove_node(u)
            if v in G:
                G.remove_node(v)

            changed = True          # re-scan after modification

    return added


def apply_degree_zero_rule(G: nx.Graph) -> int:
    """
    Remove isolated vertices — they never need to be in the cover.
    Returns count removed.
    """
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    return len(isolated)


def apply_high_degree_rule(G: nx.Graph, cover: set, k: int) -> tuple[bool, int]:
    """
    High-degree rule (Crown / LP bound):
    If any vertex v has degree > k, it MUST be in the cover.
    (Branching on it would exceed budget on the 'not-in-cover' branch.)

    Returns:
        (infeasible, added)
        infeasible=True means k was exhausted — no solution exists at this k
    """
    added = 0
    changed = True

    while changed:
        changed = False
        high = [v for v, d in G.degree() if d > k]

        for v in high:
            if v not in G:
                continue
            cover.add(v)
            added += 1
            k -= 1

            if k < 0:
                return True, added  # provably infeasible

            G.remove_node(v)
            changed = True

    return False, added


def apply_triangle_rule(G: nx.Graph, cover: set) -> int:
    """
    Degree-2 / triangle rule:
    If vertex v has degree 2 with neighbors u and w:
      - If (u, w) is an edge  → u and w must both be in cover.
        (v is covered by either; the edge (u,w) forces one of them;
         taking both is at least as good as any other choice.)
      - If (u, w) is NOT an edge → we can 'fold' v:
        Replace {v, u, w} with a single meta-vertex m.
        (This is the classic folding technique — skipped for now,
         we just return 0 and let branching handle it.)

    Returns vertices added to cover.
    """
    added = 0
    changed = True

    while changed:
        changed = False
        degree_two = [v for v, d in G.degree() if d == 2]

        for v in degree_two:
            if v not in G:
                continue
            neighbors = list(G.neighbors(v))
            if len(neighbors) != 2:
                continue

            u, w = neighbors
            if G.has_edge(u, w):
                # triangle: take u and w
                for node in (u, w):
                    if node in G:
                        cover.add(node)
                        added += 1
                        G.remove_node(node)
                if v in G:
                    G.remove_node(v)
                changed = True

    return added


def apply_all_reductions(G: nx.Graph, cover: set, k: int) -> tuple[bool, int]:
    """
    Apply all reduction rules exhaustively.

    Returns:
        (infeasible, new_k)
        infeasible=True means no vertex cover of size k exists.
    """
    apply_degree_zero_rule(G)

    total_added = 0
    changed = True

    while changed:
        changed = False

        # degree-one
        n = apply_degree_one_rule(G, cover)
        if n:
            k -= n
            total_added += n
            changed = True

        if k < 0:
            return True, k

        # high-degree
        infeasible, n = apply_high_degree_rule(G, cover, k)
        if n:
            k -= n
            total_added += n
            changed = True
        if infeasible:
            return True, k

        # triangle
        n = apply_triangle_rule(G, cover)
        if n:
            k -= n
            total_added += n
            changed = True

        if k < 0:
            return True, k

        apply_degree_zero_rule(G)

    return False, k