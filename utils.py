import numpy as np
import networkx as nx

def get_U_matrix(M):
    return np.concatenate((np.array([-np.ones(M-1)]), np.eye(M-1)))

def matprint(mat, fmt="g"):
    try:
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
            print("")
    except TypeError as e:
        print(mat)


def matwrite(mat, file, fmt="g"):
    with open(file, 'w') as fp:
        col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                fp.write(("{:"+str(col_maxes[i])+fmt+"}  ").format(y))
            fp.write("\n")


def frobenius_norm_2(matrix):
    '''
    Get matrix frobenius norm squared, so it works with cvxpy
    '''
    # TODO make this good
    accum = 0
    for row in matrix:
        for column in row:
            accum += column*column
    return accum


def create_matrix_from_nx_graph(graph: nx.Graph):
    N = len(graph.nodes())
    matrix = np.zeros((N, N))
    A = np.zeros((N, N))
    for edge in graph.edges(data=True):
        matrix[edge[0], edge[1]] = -edge[2]['value']
        matrix[edge[1], edge[0]] = -edge[2]['value']
        A[edge[1], edge[0]] = 1
        A[edge[0], edge[1]] = 1
        matrix[edge[0], edge[0]] += edge[2]['value']
        matrix[edge[1], edge[1]] += edge[2]['value']
    return matrix, A

