# Tarjan's Strongly Connected Components Algorithm

This document describes Tarjan's algorithm for finding strongly connected components
(SCCs) in a directed graph. The algorithm was published by Robert Tarjan in 1972 in
his paper "Depth-First Search and Linear Graph Algorithms" (SIAM Journal on Computing).

## Background

A **strongly connected component** (SCC) of a directed graph is a maximal set of
vertices such that there is a path from every vertex to every other vertex within
the set. Every directed graph can be partitioned into one or more SCCs.

The **condensation** of a graph is formed by collapsing each SCC into a single node.
The condensation is always a directed acyclic graph (DAG).

## Core Idea

Tarjan's algorithm exploits a fundamental property: in a depth-first search of a
directed graph, the vertices of each SCC form a contiguous subtree in the DFS tree.
The first vertex visited in an SCC (during DFS) is called the **root** of that SCC.

The algorithm identifies SCC roots by tracking, for each vertex, the earliest
vertex reachable by following DFS tree edges down and then at most one back-edge
or cross-edge to a vertex still being processed.

## Terminology

**Discovery time**: The order in which a vertex is first visited during depth-first
search. This is a logical timestamp, not actual clock time. The first vertex visited
has discovery time 0, the second has discovery time 1, and so on. Also called
"discovery index" or simply "index" in some literature.

**DFS tree**: The tree formed by the edges traversed during depth-first search. When
DFS visits a new vertex w from vertex v, the edge (v, w) becomes part of the DFS tree,
with v as the parent of w. The DFS tree spans all reachable vertices from the starting
point.

**Tree edge**: An edge (v, w) where w has not yet been visited. Following this edge
adds w as a child of v in the DFS tree.

**Back edge**: An edge (v, w) where w is an ancestor of v in the DFS tree (i.e., w is
on the current DFS path from the root to v). Back edges create cycles in the original
graph.

**Cross edge**: An edge (v, w) where w has already been visited but is not an ancestor
of v. In directed graphs, cross edges go to vertices in previously completed subtrees.
For Tarjan's algorithm, we distinguish between cross edges to vertices still on the
stack (treated similarly to back edges) versus cross edges to vertices in completed
SCCs (ignored).

**Root (of an SCC)**: The first vertex of a strongly connected component to be visited
during DFS. When the algorithm finishes processing a root, all other vertices in its
SCC will be on the stack above it.

**Subtree**: In the context of DFS, the subtree rooted at vertex v consists of v and
all vertices discovered through recursive DFS calls starting from v (i.e., v's
descendants in the DFS tree).

## Data Structures

For each vertex v, we maintain:

- **index[v]**: The order in which v was first visited (discovery time).
  Vertices are numbered 0, 1, 2, ... in visitation order.

- **lowlink[v]**: The smallest index of any vertex reachable from v through
  a path of tree edges followed by at most one back/cross edge to a vertex
  that is still on the stack. Initially set to index[v].

- **on_stack[v]**: Boolean indicating whether v is currently on the stack.
  This is crucial: we only consider edges to vertices still on the stack
  when updating lowlink, because vertices not on the stack belong to
  already-completed SCCs.

Global state:

- **index_counter**: Monotonically increasing counter for assigning indices.

- **S**: A stack of vertices. Vertices are pushed when first visited and
  popped when their SCC is identified. The stack maintains the invariant
  that a vertex remains on it if and only if there exists a path from it
  to some vertex earlier on the stack (i.e., it might be part of the same
  SCC as vertices below it).

## Why Lowlink Matters

The lowlink value is the mechanism for detecting SCC roots.

Think of it this way: when we're at vertex v, we explore its entire subtree (all the
vertices we can reach by recursing deeper). During that exploration, we might find
edges that point back to vertices we visited earlier - vertices that are still on the
stack, still being processed.

lowlink[v] records the smallest index (earliest discovery time) among:
1. v itself
2. Any vertex on the stack that v can reach directly via a back/cross edge
3. Any vertex on the stack that any descendant of v can reach

In other words: "If I explore everything below v in the DFS tree, what's the earliest
vertex still on the stack that I can get back to?"

**SCC root detection**:
- If `lowlink[v] == index[v]`, then nothing in v's subtree can "escape" to an earlier
  vertex. This means v is the root of an SCC. All vertices on the stack from v to the
  top were discovered after v and couldn't reach anything earlier - they must all be
  in the same SCC as v. We pop them all off to form one complete SCC.
- If `lowlink[v] < index[v]`, then some descendant of v found a path back to an earlier
  vertex. So v isn't a root - it's part of a larger SCC whose root is that earlier vertex.

**How lowlink propagates**: When we're at v and see an edge to w, there are three cases:

1. **w is unvisited**: We must recurse into w first. After returning, w's lowlink is
   available, and we update `lowlink[v] := min(lowlink[v], lowlink[w])`.

2. **w is on the stack**: w was already visited but hasn't been assigned to an SCC yet.
   This means w could be part of the same SCC as v. We update lowlink[v] immediately.
   Tarjan's original uses `min(lowlink[v], index[w])`; a common variant uses
   `min(lowlink[v], lowlink[w])`. Both produce correct SCCs.

3. **w was visited but is off the stack**: w belongs to an already-completed SCC.
   There's no path from w back to any vertex we're still processing, so we ignore
   this edge.

The first two cases differ in *when* we can access information about w (before vs after
recursion), but if using the variant, the update formula is the same: `min` with w's
lowlink.

**Example**: If vertices A→B→C→A form a cycle and we visit in order A, B, C:
- C discovers an edge back to A (index 0), sets lowlink[C] = 0
- B gets lowlink[B] = min(1, lowlink[C]) = 0
- A gets lowlink[A] = min(0, lowlink[B]) = 0
- A checks: lowlink[A] == index[A]? Yes (both 0), so A is the root. Pop {C, B, A} as
  one SCC.

Without lowlink, we'd have no way to know when we've found a complete SCC versus when
we're still in the middle of one.

## Algorithm

```
algorithm tarjan_scc(G):
    // Input: Directed graph G = (V, E)
    // Output: List of SCCs and the condensation graph (edges between SCCs)

    index_counter := 0          // Global counter for discovery times
    S := empty stack            // Stack of vertices being processed
    index := array of size |V|  // Discovery time for each vertex
    lowlink := array of size |V|
    on_stack := array of size |V|
    SCCs := empty list          // Accumulates the identified SCCs

    // For building the condensation graph:
    scc_id := array of size |V|     // Maps each vertex to its SCC index
    scc_count := 0                  // Counter for numbering SCCs

    // Mark all vertices as unvisited (index undefined)
    for each vertex v in V:
        index[v] := UNDEFINED

    // ─────────────────────────────────────────────────────────────────
    // PHASE 1: Find all SCCs using depth-first search
    // ─────────────────────────────────────────────────────────────────
    for each vertex v in V:
        if index[v] = UNDEFINED:
            strongconnect(v)

    // ─────────────────────────────────────────────────────────────────
    // PHASE 2: Build the condensation graph
    // ─────────────────────────────────────────────────────────────────
    // Now that every vertex has been assigned to an SCC, we iterate
    // through all edges to find which ones cross SCC boundaries.
    condensation_edges := empty set

    for each edge (u, v) in E:
        if scc_id[u] ≠ scc_id[v]:
            // This edge crosses from one SCC to another.
            // Add it to the condensation graph.
            add (scc_id[u], scc_id[v]) to condensation_edges

    return (SCCs, scc_id, condensation_edges)


function strongconnect(v):
    // Called when vertex v is first discovered.
    // Explores all vertices reachable from v and identifies any SCCs
    // rooted at v or its descendants.

    // ─────────────────────────────────────────────────────────────
    // STEP 1: Initialize this vertex
    // ─────────────────────────────────────────────────────────────
    // Assign the next available index to v. This is v's discovery time.
    index[v] := index_counter

    // Initially, the earliest reachable vertex from v is v itself.
    // This will be updated as we explore v's descendants and neighbors.
    lowlink[v] := index_counter

    index_counter := index_counter + 1

    // Push v onto the stack. It stays there until we determine which
    // SCC it belongs to. The stack contains all vertices on the current
    // DFS path, plus vertices from the current path's subtrees that
    // haven't yet been assigned to an SCC.
    push(S, v)
    on_stack[v] := true

    // ─────────────────────────────────────────────────────────────
    // STEP 2: Explore all outgoing edges from v
    // ─────────────────────────────────────────────────────────────
    for each edge (v, w) in E:
        if index[w] = UNDEFINED:
            // ─────────────────────────────────────────────────────
            // Case A: w has not been visited yet (tree edge)
            // ─────────────────────────────────────────────────────
            // w is a new vertex; recurse into it. This edge (v, w)
            // becomes a tree edge in the DFS tree.
            strongconnect(w)

            // After returning, w's lowlink reflects the earliest vertex
            // reachable from w. If w can reach an earlier vertex, then
            // so can v (via w).
            lowlink[v] := min(lowlink[v], lowlink[w])

        else if on_stack[w]:
            // ─────────────────────────────────────────────────────
            // Case B: w is on the stack (back edge or cross edge
            //         to an ancestor or vertex in current SCC)
            // ─────────────────────────────────────────────────────
            // w is already visited AND still on the stack, meaning w
            // is an ancestor of v in the current DFS path, or w is in
            // the same SCC as an ancestor. Either way, v can reach w.
            //
            // We update lowlink[v] using index[w] (not lowlink[w]).
            // This is correct because:
            // - w is already on the stack, so (v, w) is not a tree edge
            // - We want the earliest vertex v can reach directly via
            //   this single edge, which is w itself (index[w])
            // - Using lowlink[w] would be incorrect because it might
            //   reflect vertices reachable through w's subtree, but
            //   the path v -> w -> ... doesn't go through v's subtree
            lowlink[v] := min(lowlink[v], index[w])

        // ─────────────────────────────────────────────────────────
        // Case C (implicit else): w was visited but is not on the stack
        // ─────────────────────────────────────────────────────────
        // w belongs to an SCC that has already been completed. This edge
        // doesn't help find cycles (no path back from w), so we don't
        // update lowlink. The edge will be captured in Phase 2 when we
        // build the condensation graph.

    // ─────────────────────────────────────────────────────────────
    // STEP 3: Check if v is the root of an SCC
    // ─────────────────────────────────────────────────────────────
    // After exploring all of v's descendants, check if v is the root
    // of a strongly connected component.
    //
    // v is a root if lowlink[v] = index[v], meaning:
    // - No vertex in v's subtree can reach any vertex discovered
    //   before v (that is still on the stack)
    // - Therefore, v and all vertices above it on the stack (up to
    //   and including v) form a maximal strongly connected component
    //
    // Why this works:
    // - If lowlink[v] < index[v], then some vertex reachable from v
    //   can reach a vertex discovered before v, so v is not a root
    // - If lowlink[v] = index[v], then v is the earliest reachable
    //   vertex from its entire subtree, making it the SCC root

    if lowlink[v] = index[v]:
        // v is the root of an SCC. Pop all vertices from the stack
        // up to and including v. These vertices form the SCC.
        //
        // All these vertices were pushed after v (or are v itself)
        // and couldn't reach any vertex earlier than v, so they
        // must all be in the same SCC as v.

        current_scc := scc_count
        scc_count := scc_count + 1

        SCC := empty list
        repeat:
            w := pop(S)
            on_stack[w] := false
            scc_id[w] := current_scc    // Record which SCC this vertex belongs to
            append w to SCC
        until w = v

        // Record this SCC
        append SCC to SCCs
```

## Complexity Analysis

**Time Complexity: O(V + E)**

- Phase 1 (SCC detection):
  - Each vertex is visited exactly once by strongconnect (guarded by the index check)
  - Each edge is examined exactly once (when exploring from its source vertex)
  - Stack operations (push/pop) are O(1) and each vertex is pushed and popped once
  - All lowlink updates are O(1)
- Phase 2 (condensation construction):
  - Each edge is examined exactly once to check if it crosses SCC boundaries

**Space Complexity: O(V + E')**

- The index, lowlink, on_stack, and scc_id arrays each require O(V) space
- The stack S contains at most V vertices
- The recursion depth is at most V (for a path graph)
- The output SCCs collectively contain all V vertices
- The condensation_edges set contains at most E' edges, where E' ≤ E is the number
  of edges that cross SCC boundaries (intra-SCC edges don't appear)

## Important Properties

1. **Reverse Topological Order**: SCCs are identified in reverse topological order
   of the condensation DAG. The first SCC found (SCC 0) has no outgoing edges to
   other SCCs; the last SCC found might have edges to all previously found SCCs.
   This means the SCC indices are already a valid topological ordering of the
   condensation (in reverse).

2. **Single DFS**: Unlike Kosaraju's algorithm (which requires two DFS passes),
   Tarjan's algorithm finds all SCCs in a single DFS traversal. Building the
   condensation requires one additional pass over all edges.

3. **Stack Invariant**: A vertex remains on the stack if and only if it has a
   path to some vertex lower on the stack. This is the invariant that makes the
   algorithm work.

4. **Condensation Edges**: Every edge in the condensation goes from a higher-numbered
   SCC to a lower-numbered SCC. This is because if edge (u, v) exists and v's SCC
   was completed before u's SCC, then scc_id[v] < scc_id[u]. This confirms the
   reverse topological order property.

## Variation Note

Some implementations update lowlink using `min(lowlink[v], lowlink[w])` instead
of `min(lowlink[v], index[w])` when w is on the stack. Both produce correct SCCs,
but using `index[w]` is Tarjan's original formulation and has a cleaner invariant:
lowlink[v] represents the minimum index reachable, not the minimum lowlink.

## References

- Tarjan, R. E. (1972). "Depth-First Search and Linear Graph Algorithms".
  SIAM Journal on Computing, 1(2), 146-160. DOI: 10.1137/0201010

- [Wikipedia: Tarjan's strongly connected components algorithm](https://en.wikipedia.org/wiki/Tarjan's_strongly_connected_components_algorithm)

- [CP-Algorithms: Strongly Connected Components](https://cp-algorithms.com/graph/strongly-connected-components.html)

- [GeeksforGeeks: Tarjan's Algorithm](https://www.geeksforgeeks.org/dsa/tarjan-algorithm-find-strongly-connected-components/)

- [Rosetta Code: Tarjan](https://rosettacode.org/wiki/Tarjan)

- [Baeldung: Tarjan's Algorithm](https://www.baeldung.com/cs/scc-tarjans-algorithm)
