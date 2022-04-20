# If this turns out to be too difficult, we should use NetworkX
# https://networkx.org/

import random


def generate_random_small_world_graph_connected(N: int, K: int, beta: float, times: int = 100):
    i = 0
    while i < times:
        i += 1
        graph = generate_random_small_world_graph(N, K, beta)
        if is_connected_graph(graph, N):
            return graph
    return None


def generate_random_small_world_graph(N: int, K: int, beta: float):
    """
    We generate a random graph from Wattz - Strogatz model
    The algorithm used is based on its wikipedia's entry
    https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model#Algorithm
    We output an undirected graph with N nodes and NK/2 edges
    Args:
        N (int): number of nodes
        K (int): mean degree. Should be even
        beta (float): parameter 0 < beta < 1, chance to rewire

        We assume N >> K >> ln N >> 1
    """

    def circular_distance(a, b, N):
        return min((abs(a - b), N - abs(a - b)))
    # First step: Constructing regular ring lattice
    # The graph will be defined only by its adjacency list
    nodes = list(range(N))
    graph = []
    for i in range(N):
        adjacents = []
        for j in range(K // 2):
            adjacents.append((i + (j + 1)) % N)
            adjacents.append((i - (j + 1)) % N)
        graph.append(adjacents)

    # Second step: rewire
    for i in range(N):
        for j in graph[i]:
            if K // 2 > circular_distance(i, j, N):  # Not one of the regular edges
                continue
            if random.random() < beta:
                # Rewire
                adj_to_this_node: list[int] = graph[i]
                regular_graph_nodes: list[int] = [x for x in graph[i] if K // 2 >= circular_distance(i, x, N)]
                if not regular_graph_nodes:
                    continue
                edge_to_remove = random.choice(regular_graph_nodes)
                edge_to_add = random.choice([x for x in nodes if K // 2 < circular_distance(i, x, N) and x not in adj_to_this_node])

                # remove
                adj_to_edge_to_remove: list[int] = graph[edge_to_remove]
                del adj_to_edge_to_remove[adj_to_edge_to_remove.index(i)]
                del adj_to_this_node[adj_to_this_node.index(edge_to_remove)]

                # add
                graph[i].append(edge_to_add)
                graph[edge_to_add].append(i)
    return graph


def is_connected_graph(graph: list, N: int):
    # we make a dfs and then check if that found every node in the graph
    # note that the graph is an undirected one
    nodes_to_travel_through = [0]
    marked_nodes = set([0])
    while nodes_to_travel_through:
        current_node_number = nodes_to_travel_through.pop()
        adjacent_nodes = graph[current_node_number]
        for node in adjacent_nodes:
            if node not in marked_nodes:
                marked_nodes.add(node)
                nodes_to_travel_through.append(node)

    return len(marked_nodes) == N


if __name__ == '__main__':
    print(generate_random_small_world_graph_connected(14, 2, 0.1))
