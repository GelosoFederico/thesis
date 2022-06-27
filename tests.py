from NetworkMatrix import IEEE14_b_matrix
from utils import create_matrix_from_nx_graph, get_U_matrix, matprint
import numpy as np
import networkx as nx
import simulations

def test_stack_matrix():
    matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])

    vector = simulations.stack_matrix(matrix)

    stacked_vector = np.array([[1,4,7,2,5,8,3,6,9]])
    assert not (vector-stacked_vector).any()

def test_f_score_is_1():
    matrix1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    matrix2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    f_score = simulations.F_score(matrix1, matrix2)

    assert f_score == 1

def test_f_score_is_0():
    matrix1 = np.array([[0,1,0],[1,0,0],[0,0,0]])
    matrix2 = np.array([[0,0,1],[0,0,0],[1,0,0]])
    f_score = simulations.F_score(matrix1, matrix2)

    assert f_score == 0

def test_f_score_is_two_thirds():
    matrix1 = np.array([[0,1,0],[1,0,0],[0,0,0]])
    matrix2 = np.array([[0,1,1],[1,0,0],[1,0,0]])
    f_score = simulations.F_score(matrix1, matrix2)

    assert f_score == 2/3

def test_mse_same_matrix():
    matrix1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    matrix2 = np.array([[1,2,3],[4,5,6],[7,8,9]])

    mse = simulations.MSE_matrix(matrix1, matrix2)

    assert mse == 0

def test_u_matrix_equation_5():
    B, _ = IEEE14_b_matrix()
    U = get_U_matrix(14)
    B_tilde = B[1:,1:]
    threshold = min(np.diag(B))

    assert (np.abs(B - (U @ B_tilde @ U.T)) < threshold/10).all()

def test_matrix_from_graph():
    g: nx.Graph = nx.empty_graph(range(4))
    g.add_edge(0,1, value=10)
    g.add_edge(2,3, value=1)
    matrix = create_matrix_from_nx_graph(g)
    test_m = np.array([[10,-10,0,0],[-10,10,0,0],[0,0,1,-1],[0,0,-1,1]])
    threshold = min(np.diag(test_m))
    assert ((matrix - test_m) < threshold/10).all()
