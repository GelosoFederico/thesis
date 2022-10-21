# TODO recheck conditions for values in rt and cluster
# TODO assing necesary values when we create the random nt
# TODO recheck probability changes from tiednets added constraints during the algorithm
# TODO check if values like clustering are correct

import logging

# from utils import create_matrix_from_nx_graph

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from typing import List, Tuple
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_random_rt_nested_network(N: int, K: int, d: int, alpha: float, beta: float, p_rewire: float, N_subnetworks: int, distribution_params: Tuple[float, float, float]):

    # Select size of the network acording to connectivity limitation (23)
    if N > 30 and K <= 3:
        raise Exception("For k between 2 and 3, N should be less than 30")
    if N > 300 and K <= 5:
        raise Exception("For k between 4 and 5, N should be less than 300")

    # logger.info(locals())
    # Create subnetworks
    subnetworks = [generate_random_cluster_small_world_network(N, K, d, alpha, beta, p_rewire) for x in range(N_subnetworks)]

    # Fix numbers in subnet to make them part of the network
    last_subnet_num = 0
    subnets_offsets = []
    for subnet in subnetworks:
        subnet_map = {x:x+last_subnet_num for x in range(len(subnet))}
        nx.relabel_nodes(subnet, subnet_map,copy=False)

        subnets_offsets.append(last_subnet_num)
        last_subnet_num += len(subnet)
    complete_length = sum((len(x) for x in subnetworks))
    complete_graph: nx.Graph = nx.empty_graph(range(complete_length))
    for subnet in subnetworks:
        for edge in subnet.edges(data=True):
            complete_graph.add_edge(edge[0], edge[1], group=edge[2]['group'])

    # Join subnets
    for i, subnet in enumerate(subnetworks):
        subnet_before = subnetworks[i-1]
        join_subnets_at_random(complete_graph, subnet, subnet_before, K, subnets_offsets[i], subnets_offsets[i-1])

    # Assign impedance values
    # First we generate them
    # TODO neither scipy nor numpy have the double pareto distribution, nor they can clip them.
    # So we use lognormal and clip them ourselves.
    # TODO send distribution as parameter to this function
    impedance_values = []
    for edge in complete_graph.edges():
        found = False
        while not found:
            value_candidate = np.random.lognormal(distribution_params[0], distribution_params[1])
            if value_candidate < distribution_params[2]:
                impedance_values.append(value_candidate)
                found = True
    impedance_values.sort()
    lattice_conn = [edge for edge in complete_graph.edges(data=True) if edge[2]['group'] == 'lattice-conn']
    rewire = [edge for edge in complete_graph.edges(data=True) if edge[2]['group'] == 'rewire']
    local = [edge for edge in complete_graph.edges(data=True) if edge[2]['group'] == 'local']
    random.shuffle(lattice_conn)
    random.shuffle(rewire)
    random.shuffle(local)
    all_edges = lattice_conn + rewire + local
    for edge in all_edges:
        nx.set_edge_attributes(complete_graph, {(edge[0], edge[1]): {'value': impedance_values.pop()}})
    return complete_graph


def join_subnets_at_random(complete_graph: nx.Graph,
                           net1: nx.Graph,
                           net2: nx.Graph,
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
        if not complete_graph.has_edge(choice1_real, choice2_real):
            complete_graph.add_edge(choice1_real, choice2_real, group='lattice-conn')
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


def generate_random_cluster_small_world_network(N: int,
                                                K: int,
                                                d: int,
                                                alpha: float,
                                                beta: float,
                                                p_rewire: float,
                                                max_tries: int = 100) -> nx.Graph:
    connected = False
    tries = 0
    while not connected:
        graph = _generate_random_cluster_small_world_network(N, K, d, alpha, beta, p_rewire)
        logger.debug(f"finished cluster try: {tries}")
        connected = nx.is_connected(graph)
        tries += 1
        if tries > max_tries:
            raise Exception("It took more times than allowed to make a connected random cluster small world network")
        if not connected:
            logger.info("Graph not connected. Restarting")
    return graph


def _generate_random_cluster_small_world_network(N: int, K: int, d: int, alpha: float, beta: float, p_rewire: float) -> nx.Graph:
    created_graph: nx.Graph = nx.empty_graph(range(N))
    # We select the number of edges to generate for each neighborhood around each node
    all_ks = []
    for i in range(N):
        found = False
        while not found:
            new_contender = np.random.default_rng().geometric(p=1/K)
            if new_contender < 2 * d:
                found = True
                all_ks.append(new_contender)
    # as we are not getting a real geometric distribution, we will print the curren expectance
    p = 1 / K
    real_k = sum((x * p * np.power((1 - p), x - 1) for x in range(2 * d))) / (1 - np.power(1 - p, 2))
    logger.debug(f"Real K is {real_k:.2f}")
    logger.debug("Starting link selection")
    for i in range(N):
        k_node = all_ks[i]
        possible_nodes = [x for x in range(i - d, i + d + 1)]  # Nodes in the neighborhood
        possible_nodes = [x % N for x in possible_nodes if x != i]  # Except itself, and use modulo to be circular
        np.random.shuffle(possible_nodes)
        nodes_to_add_edges = possible_nodes[:k_node]
        for to_edge in nodes_to_add_edges:
            created_graph.add_edge(i, to_edge, group='local')
    logger.debug(f"Graph is now {created_graph}")

    # rewiring, literal from tiedNets
    # Markov chain
    prev_value = 1  # Nowhere in the paper explicitly says it starts at 1.
    markov_results = []
    logger.debug("Starting Markov chain")
    for i in range(N):
        rand_value = random.random()
        if prev_value == 0 and rand_value < alpha:
            markov_results.append(1)
        elif prev_value == 1 and rand_value < beta:
            markov_results.append(0)
        else:
            markov_results.append(prev_value)
        prev_value = markov_results[-1]
    logger.debug(f"Markov chain results are {markov_results}")

    # We make clusters from those that are adjacent with the same markov result
    clusters = []
    starts_with_cluster = False
    ends_with_cluster = False
    in_cluster = False
    logger.debug("Creating clusters")
    for i, v in enumerate(markov_results):
        if v == 1:
            if not in_cluster:
                in_cluster = True
                clusters.append([])
            if i == 0:
                starts_with_cluster = True
            if i == N-1:
                ends_with_cluster = True
            clusters[-1].append(i)
        else:
            in_cluster = False
    if starts_with_cluster and ends_with_cluster and len(clusters) > 1:
        for elem in clusters[-1]:
            clusters[0].append(elem)
        clusters.pop()
    logger.debug(f"Clusters are {clusters}")
    # They would restart if there is only one cluster to rewire
    logger.debug("Starting to rewire between clusters")
    if len(clusters) > 2:
        for cluster in clusters:
            other_clusters = [x for x in clusters if x != cluster]
            possible_nodes_in_other_clusters = []
            for node in cluster:
                for cluster in other_clusters:
                    for other_node in cluster:
                        if not created_graph.has_edge(node, other_node):
                            possible_nodes_in_other_clusters.append(other_node)
                to_delete = []
                to_add = []
                for edge in created_graph.edges(node):
                    other_node = [x for x in edge if x != node][0]
                    random_number = random.random()
                    if other_node in cluster and random_number < p_rewire:
                        to_delete.append((node, other_node))
                        new_edge = random.choice(possible_nodes_in_other_clusters)
                        to_add.append((node, new_edge))
                        logger.debug(f"Rewiring in node {node} from {edge} to {new_edge}")
                    else:
                        logger.debug(f"Not rewiring in node {node} from {edge} with values {random_number}, {other_node in cluster}")

                for edge in to_delete:
                    created_graph.remove_edge(edge[0], edge[1])
                for edge in to_add:
                    created_graph.add_edge(edge[0], edge[1], group='rewire')

    logger.debug("Finished rewiring")
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


def dpln_distribution(alpha: float, beta: float, nu: float, tau: float, size: np.matrix=None):
    """
    Definition based on
    The Double Pareto-Lognormal Distribution – A New Parametric Model for Size Distributions, 
    William J. Reed and Murray Jorgensen, 2000

    We obtain X as X = U*V_1/V_2
    where
    log(U) ~ N(nu, tau**2)
    V_1 ~ pareto(alpha)
    V_2 ~ pareto(beta)

    TODO check if this pareto is the same used in the paper
    """
    lognormal = np.random.lognormal(nu, tau, size)
    pareto_1 = np.random.pareto(alpha, size)
    pareto_2 = np.random.pareto(beta, size)

    return lognormal * pareto_1 / pareto_2


def get_matrix_from_nt_graph(G: nx.Graph):
    for edge in G.edges():
        # TODO changed for every value to be 1
        G[edge[0]][edge[1]]['weight'] = G[edge[0]][edge[1]]['value']
    matrix = nx.to_numpy_matrix(G)
    # print(matrix.shape)
    # print(matrix.shape[0])
    # for i in range(matrix.shape[0]):
    #     print(i)
    #     # print(matrix[0][0])
    #     # print(matrix)
    #     matrix[i,i] = 1 # sum(matrix[i])
    # print(matrix)
    return matrix


if __name__ == '__main__':
    n_nodes = 14
    n_subnets = 4
    # le_graph = generate_random_rt_nested_network(n_nodes, 2, 2, 0.6, 0.6, 0.5, n_subnets)
    le_graph = generate_random_rt_nested_network(n_nodes, 2, 4, 0.5, 0.4, 0.8, n_subnets, (-2.4, 2.1, 2.0))
    for edge in le_graph.edges(data=True):
        print(edge)
    matr = get_matrix_from_nt_graph(le_graph)
    # print(le_graph)
    # # print(generate_random_cluster_small_world_network(14, 2, 4))
    # G = nx.empty_graph(range(n_nodes * n_subnets))
    # last_subnet_num = 0
    # for subnet in le_graph:
    #     for i, node in enumerate(subnet):
    #         pos = i + last_subnet_num
    #         for edge in node:
    #             if not G.has_edge(pos, edge):
    #                 G.add_edge(pos, edge)
    #     last_subnet_num += len(subnet)
    nx.draw_circular(le_graph, with_labels=True)  # TODO draw edges with different colors separating by group
    plt.show()
    # mat = create_matrix_from_nx_graph(le_graph)
    # plt.matshow(mat)
    # plt.show()
