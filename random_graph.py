# If this turns out to be too difficult, we should use NetworkX
# https://networkx.org/

import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from typing import List
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_random_rt_nested_network(N: int, K: int, d: int, alpha:float, beta: float, p_rewire: float, N_subnetworks: int):
    # Select size of the network acording to connectivity limitation (23)
    # 1 - crea
    if N > 30 and K <= 3:
        raise Exception("For k between 2 and 3, N should be less than 30")
    if N > 300 and K <= 5:
        raise Exception("For k between 4 and 5, N should be less than 300")

    # Create subnetworks
    subnetworks = [generate_random_cluster_small_world_network(N, K, d, alpha, beta, p_rewire) for x in range(N_subnetworks)]

    # Fix numbers in subnet to make them part of the network
    last_subnet_num = 0
    subnets_offsets = []
    for subnet in subnetworks:
        for edges_list in subnet:
            for i, edge in enumerate(edges_list):
                edges_list[i] += last_subnet_num

        subnets_offsets.append(last_subnet_num)
        last_subnet_num += len(subnet)

    # print()
    # Join them
    for i, subnet in enumerate(subnetworks):
        subnet_before = subnetworks[i-1]
        join_subnets_at_random(subnet, subnet_before, K, subnets_offsets[i], subnets_offsets[i-1])

    return subnetworks


def join_subnets_at_random(net1: List[List[int]],
                           net2: List[List[int]],
                           K: int,
                           net1_offset: int,
                           net2_offset: int):
    tries = 0
    connections = 0
    while connections < K and tries < 100:
        choice1 = random.choice(range(len(net1)))
        choice2 = random.choice(range(len(net2)))
        choice1_real = choice1 + net1_offset
        choice2_real = choice2 + net2_offset
        if choice2_real not in net1[choice1] and choice1_real not in net2[choice2]:
            net1[choice1].append(choice2_real)
            net2[choice2].append(choice1_real)
            connections += 1
        tries += 1


def add_edge(graph:List[List], node_from:int, node_to:int):
    if node_to not in graph[node_from]:
        graph[node_from].append(node_to)
    if node_from not in graph[node_to]:
        graph[node_to].append(node_from)


def remove_edge(graph:List[List], node_from:int, node_to:int):
    if node_to in graph[node_from]:
        graph[node_from].remove(node_to)
    if node_from in graph[node_to]:
        graph[node_to].remove(node_from)


def generate_random_cluster_small_world_network(N: int, K: int, d: int, alpha: float, beta: float, p_rewire: float, max_tries: int=100):
    connected = False
    tries = 0
    while not connected:
        graph = _generate_random_cluster_small_world_network(N, K, d, alpha, beta, p_rewire)
        connected = is_connected_graph(graph, N)
        tries += 1
        if tries > max_tries:
            raise Exception("It took more times than allowed to make a connected random cluster small world network")
        if not connected:
            logger.info("Graph not connected. Restarting")
    return graph


def _generate_random_cluster_small_world_network(N: int, K: int, d: int, alpha: float, beta: float, p_rewire: float):
    created_graph = []
    for i in range(N):
        created_graph.append([])
    # We select the number of edges to generate for each neighborhood around each node
    all_ks = np.random.default_rng().geometric(p=1/K, size=N)
    all_ks = [k if k < 2*d else 2*d for k in all_ks]  # In network they would fail if it tries to go further than 2d
    logger.info("Starting link selection")
    for i in range(N):
        k_node = all_ks[i]
        possible_nodes = [x for x in range(i-d,i+d+1)]
        possible_nodes = [x % N for x in possible_nodes if x != i]
        np.random.shuffle(possible_nodes)
        nodes_to_add_edges = possible_nodes[:k_node]
        for to_edge in nodes_to_add_edges:
            add_edge(created_graph, i, to_edge)
    logger.info(f"Graph is now {created_graph}")

    # rewiring, literal from tiedNets
    # Markov chain
    prev_value = 1
    markov_results = []
    logger.info("Starting Markov chain")
    for i in range(N):
        rand_value = random.random()
        if prev_value == 0 and rand_value < alpha:
            markov_results.append(1)
        elif prev_value == 1 and rand_value < beta:
            markov_results.append(0)
        else:
            markov_results.append(prev_value)
        prev_value = markov_results[-1]
    logger.info(f"Markov chain results are {markov_results}")

    # We make clusters from those that are adjacent with the same markov result
    clusters = []
    starts_with_cluster = False
    ends_with_cluster = False
    in_cluster = False
    logger.info("Creating clusters")
    for i, v in enumerate(markov_results):
        if v == 1:
            if not in_cluster:
                in_cluster = True
                clusters.append([])
            if i == 0:
                starts_with_cluster = True
            if i == N:
                ends_with_cluster = True
            clusters[-1].append(i)
        else:
            in_cluster = False
    if starts_with_cluster and ends_with_cluster and len(clusters) > 1:
        for elem in clusters[-1]:
            clusters[0].append(elem)
        clusters.pop()
    logger.info(f"Clusters are {clusters}")
    # They would restart if there is only one cluster to rewire
    logger.info("Starting to rewire between clusters")
    if len(clusters) > 2:
        for cluster in clusters:
            other_clusters = [x for x in clusters if x != cluster]
            possible_nodes_in_other_clusters = []
            for node in cluster:
                for cluster in other_clusters:
                    for other_node in cluster:
                        if other_node not in created_graph[node]:
                            possible_nodes_in_other_clusters.append(other_node)
                for edge in created_graph[node]:
                    # edge is the other node to which this points to
                    if edge in cluster and random.random() < p_rewire:
                        remove_edge(created_graph, node, edge)
                        new_edge = random.choice(possible_nodes_in_other_clusters)
                        add_edge(created_graph, node, new_edge)
                        logger.info(f"Rewiring in node {node} from {edge} to {new_edge}")
    logger.info("Finished rewiring")






    return created_graph


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
    n_nodes = 14
    n_subnets = 4
    le_graph = generate_random_rt_nested_network(n_nodes, 2, 2, 0.6, 0.6, 0.5, n_subnets)
    print(le_graph)
    # print(generate_random_cluster_small_world_network(14, 2, 4))
    G = nx.empty_graph(range(n_nodes * n_subnets))
    last_subnet_num = 0
    for subnet in le_graph:
        for i, node in enumerate(subnet):
            pos = i + last_subnet_num
            for edge in node:
                if not G.has_edge(pos, edge):
                    G.add_edge(pos, edge)
        last_subnet_num += len(subnet)
    nx.draw_circular(G, with_labels=True)
    plt.show()