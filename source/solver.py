"""
Branch-and-reduce MVC solver.

Strategy:
  1. Apply reduction rules to shrink the graph.
  2. If graph is empty → solution found.
  3. Pick a branching vertex (highest degree).
  4. Branch: vertex IN cover  vs.  vertex NOT IN cover.
  5. Recurse with k decremented accordingly.

Tracks:
  - nodes_visited  : search tree nodes explored
  - best_cover     : smallest cover found so far
"""

import copy
import networkx as nx
from reductions import apply_all_reductions


class MVCSolverNaive:

    def __init__(self):
        self.nodes_visited: int = 0
        self.best_cover: set | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, G: nx.Graph) -> tuple[set, int]:
        """
        Find a minimum vertex cover of G.

        Returns:
            (cover_set, nodes_visited)
        """
        self.nodes_visited = 0
        self.best_cover = None

        n = G.number_of_nodes()

        # Binary search on k (cover size) from 0 upward.
        # In practice for MVC we want the smallest k that succeeds.
        for k in range(n + 1):
            self.nodes_visited = 0
            cover = set()
            working_graph = G.copy()

            success = self._branch(working_graph, cover, k)
            if success:
                self.best_cover = cover
                return cover, self.nodes_visited

        # Should never reach here for a valid graph
        return set(G.nodes()), self.nodes_visited

    # ------------------------------------------------------------------
    # Recursive branch-and-reduce
    # ------------------------------------------------------------------

    def _branch(self, G: nx.Graph, cover: set, k: int) -> bool:
        """
        Returns True if a vertex cover of size ≤ k exists in G,
        adding chosen vertices to `cover`.
        """
        self.nodes_visited += 1

        # --- Reduction phase ---
        infeasible, k = apply_all_reductions(G, cover, k)

        if infeasible:
            return False

        # --- Base cases ---
        if G.number_of_edges() == 0:
            return True     # all edges covered

        if k <= 0:
            return False    # budget exhausted, edges remain

        # --- Pick branching vertex: highest degree ---
        branch_vertex = max(G.degree(), key=lambda x: x[1])[0]

        # Branch 1: branch_vertex IS in the cover
        G1 = G.copy()
        cover1 = set()
        G1.remove_node(branch_vertex)
        cover1.add(branch_vertex)
        if self._branch(G1, cover1, k - 1):
            cover.update(cover1)
            return True

        # Branch 2: branch_vertex is NOT in cover
        #   → all its neighbors MUST be in the cover
        G2 = G.copy()
        cover2 = set()
        neighbors = list(G2.neighbors(branch_vertex))
        cover2.update(neighbors)
        new_k = k - len(neighbors)

        if new_k >= 0:
            for nb in neighbors:
                if nb in G2:
                    G2.remove_node(nb)
            if branch_vertex in G2:
                G2.remove_node(branch_vertex)

            if self._branch(G2, cover2, new_k):
                cover.update(cover2)
                return True

        return False

    # ------------------------------------------------------------------
    # Component-aware solving (paper's key contribution)
    # ------------------------------------------------------------------
class MVCSolverComponentAware:

    def __init__(self):
        self.nodes_visited: int = 0
        self.best_cover: set | None = None
    
    def solve(self, G: nx.Graph) -> tuple[set,int]:
        self.nodes_visited = 0
        self.best_cover = None

        n = G.number_of_nodes()

        for k in range(n+1):
            self.nodes_visited = 0
            cover = set()
            working_graph = G.copy()

            success=self._branch(working_graph,cover,k)
            if success:
                self.best_cover = cover
                return cover, self.nodes_visited
        return set(G.nodes()), self.nodes_visited
    def _branch(self,G: nx.Graph, cover: set, k: int)->bool:
        self.nodes_visited += 1

        infeasible,k = apply_all_reductions(G,cover,k)

        if infeasible:
            return False
        if G.number_of_edges()==0:
            return True
        if k<=0: 
            return False
        
        components = list(nx.connected_components(G))
        if len(components)>1:
            return self._solve_components(G,cover,k,components)
        
        branch_vertex = max(G.degree(), key = lambda x:x[1])[0]

        G1 = G.copy()
        cover1 = set()
        G1.remove_node(branch_vertex)
        cover1.add(branch_vertex)
        if self._branch(G1, cover1, k - 1):
            cover.update(cover1)
            return True

        G2 = G.copy()
        cover2 = set()
        neighbors = list(G2.neighbors(branch_vertex))
        cover2.update(neighbors)
        new_k = k - len(neighbors)
 
        if new_k >= 0:
            for nb in neighbors:
                if nb in G2:
                    G2.remove_node(nb)
            if branch_vertex in G2:
                G2.remove_node(branch_vertex)
 
            if self._branch(G2, cover2, new_k):
                cover.update(cover2)
                return True
 
        return False
 


    def _solve_components(
        self,
        G: nx.Graph,
        cover: set,
        k: int,
        components: list[set],
    ) -> bool:
        """
        When the graph splits into disconnected components,
        solve each independently and sum their cover sizes.

        This avoids re-exploring the same sub-problems together,
        which is the central insight of the paper.
        """
        # Sort components: smallest first (fail fast on infeasible ones)
        components_sorted = sorted(components, key=len)

        return self._solve_next_component(
            G, cover, k, components_sorted, index=0
        )

    def _solve_next_component(
        self,
        G: nx.Graph,
        cover: set,
        remaining_k: int,
        components: list[set],
        index: int,
    ) -> bool:
        """
        Recursively solve components one at a time,
        distributing the remaining k budget across them.
        """
        if index == len(components):
            return True     # all components solved

        component_nodes = components[index]
        subgraph = G.subgraph(component_nodes).copy()
        num_nodes = len(component_nodes)

        # Try each possible allocation of k_i to this component
        # k_i ranges from 0 to min(remaining_k, component_size)
        for k_i in range(min(remaining_k, num_nodes) + 1):
            sub_cover = set()
            sub_graph_copy = subgraph.copy()

            if self._branch(sub_graph_copy, sub_cover, k_i):
                # This component solved with k_i vertices
                # Recurse on remaining components with updated budget
                full_cover_extension = set()
                if self._solve_next_component(
                    G,
                    full_cover_extension,
                    remaining_k - k_i,
                    components,
                    index + 1,
                ):
                    cover.update(sub_cover)
                    cover.update(full_cover_extension)
                    return True

        return False    # no valid allocation found
    
MVCSolver = MVCSolverComponentAware