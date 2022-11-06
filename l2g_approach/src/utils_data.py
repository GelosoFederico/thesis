
import random
from numpy import ndarray
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split

import networkx as nx
import scipy
import pickle
import multiprocess
from functools import partial

from src.utils import *
from src.random_graph_rt_nested import generate_random_rt_nested_network, get_matrix_from_nt_graph

import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


#%%
def data_loading(dir_dataset, batch_size = None, train_prop=0.8):

    with open(dir_dataset, 'rb') as handle:
        dataset = pickle.load(handle)

    print('loading data at ', dir_dataset)

    num_samples = len(dataset['z'])
    w = [squareform(dataset['W'][i].A) for i in range(num_samples)]

    # old test_size
    # test_size = 64
    # num_samples -= 64

    # In order to assure everything fits in the batch size, we will make them according to that
    test_size = batch_size * 2
    num_samples -= test_size
    train_size = int(train_prop * num_samples)
    val_size = int(num_samples - train_size)

    # data = TensorDataset(torch.Tensor(dataset['z'][:train_size + val_size + test_size]), torch.Tensor(w[:train_size + val_size + test_size]))
    data = TensorDataset(torch.Tensor(dataset['z']), torch.Tensor(w))
    # train_data, val_data, test_data, _ = random_split(data, [train_size, val_size, test_size, extra_size])
    # print(f"train_data: {train_data},val_data: {val_data},test_data: {test_data},_: {_},")
    print(len(data))
    print(f"train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")
    print(train_size + val_size + test_size)
    train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])

    if batch_size is not None:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=test_size, shuffle=True)
        print('successfully loading: train {}, val {}, test {}, batch {}'.format(train_size, val_size,
                                                                                 test_size, batch_size))
        return train_loader, val_loader, test_loader

    else:
        print('successfully loading: train size {}, val size {}, test size {}'.format(train_size, val_size, test_size))
        return train_data, val_data, test_data

#%%

def test_data_loading(dir_dataset):

    with open(dir_dataset, 'rb') as handle:
        dataset = pickle.load(handle)

    print('loading data at ', dir_dataset)

    num_samples = len(dataset['z'])
    w = [squareform(dataset['W'][i].A) for i in range(num_samples)]

    test_size = 64
    test_data = TensorDataset(torch.Tensor(dataset['z']), torch.Tensor(w))
    test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)

    return test_loader

#%%

def _generate_BA_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, weight_scale = False):

    G = nx.barabasi_albert_graph(num_nodes, graph_hyper)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (1e-04) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def generate_BA_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, weight_scale):
    n_cpu = multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_BA_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result

#%%


def rotate_matrix(matrix):
    random_pos = list(range(len(matrix)))
    random.shuffle(random_pos)
    for from_p, to in enumerate(random_pos):
        # Col
        temp = matrix[to,:].copy()
        matrix[to,:]  = matrix[from_p,:]
        matrix[from_p,:] =  temp
        # Row
        temp = matrix[:,to].copy()
        matrix[:,to]  = matrix[:,from_p]
        matrix[:,from_p] = temp

    return matrix

def _generate_WS_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, weight_scale = False):

    G = nx.watts_strogatz_graph(num_nodes, k = graph_hyper['k'], p = graph_hyper['p'])

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)
    # W_GT *= 10

    W_GT = rotate_matrix(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (1e-04) * np.eye(num_nodes))
    #signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    #z = get_distance_halfvector(signal)

    # signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def _generate_nested_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, weight_scale = False):


    # raise Exception()
    # TODO change this and the weights with our generator
    default_hyper = {
        'k':2,
        'd':4,
        'alpha':0.5,
        'beta':0.4,
        'p_rewire':0.8, 
        'n_subnets': 4,
        'distribution_params': (-2.4, 2.1, 2.0),
    }
    # print(graph_hyper)
    # print(default_hyper)
    rt_parameters = {**default_hyper, **graph_hyper}
    # for key, value in default_hyper.items():
    #     if rt_parameters[key] != value:
    #         print(f"Using non default value {rt_parameters[key]} instead of {value} for {key}")
    # subnet_nodes_num = round(num_nodes / rt_parameters['n_subnets'])
    # logger.debug(f"subnet_nodes_num {subnet_nodes_num}")
    logger.debug(f"num_nodes {num_nodes}")

    G = None
    while not G:
        try:
            G = generate_random_rt_nested_network(
                num_nodes,
                K=rt_parameters['k'],
                d=rt_parameters['d'],
                alpha=rt_parameters['alpha'],
                beta=rt_parameters['beta'],
                p_rewire=rt_parameters['p_rewire'],
                N_subnetworks=rt_parameters['n_subnets'],
                distribution_params=rt_parameters['distribution_params'],
            )
        except Exception as e:
            logger.info(f"Exception {e} while creating graph")
    # subnet_nodes_num = round(num_nodes / rt_parameters['n_subnets'])
    # real_n_nodes = subnet_nodes_num * rt_parameters['n_subnets']
    # logger.debug(f"real_n_nodes {subnet_nodes_num}")
    logger.debug(f"num_nodes {num_nodes}")

    # G = nx.watts_strogatz_graph(num_nodes, k = graph_hyper['k'], p = graph_hyper['p'] )

    W_GT = get_matrix_from_nt_graph(G)  # TODO this should be the one we are using
    # W_GT = W_GT * 100

    # W_GT_2 = nx.adjacency_matrix(G).A
    # weights = np.random.lognormal(0, 0.1, (real_n_nodes, real_n_nodes))
    # weights = (weights + weights.T) / 2
    # W_GT_2 = W_GT_2 * weights

    W_GT = rotate_matrix(W_GT)


    # if weight_scale:
    #     W_GT = W_GT * num_nodes / np.sum(W_GT)

    return get_z_and_w_gt(W_GT, num_nodes, num_signals)


def get_z_and_w_gt(W_GT: ndarray, num_nodes: int, num_signals: int, A=None):
    if A is None:
        A = W_GT.getA()
    L_GT = np.diag(A @ np.ones(num_nodes)) - A
    W_GT = scipy.sparse.csr_matrix(W_GT)
    cov = np.linalg.inv(L_GT + (1e-04) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))
    return z, W_GT

def get_z_and_w_gt_ndarray(W_GT: ndarray, num_nodes: int, num_signals: int):
    L_GT = np.diag(W_GT.A @ np.ones(num_nodes)) - W_GT.A
    W_GT = scipy.sparse.csr_matrix(W_GT)
    cov = np.linalg.inv(L_GT + (1e-04) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))
    return z, W_GT


def generate_WS_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, weight_scale) -> dict:
    logger.info("generating WS graphs")
    n_cpu = 1 #multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_WS_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result
def generate_rt_nested_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, weight_scale) -> dict:
    logger.info("generating nested graphs")
    n_cpu = 1 # multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_nested_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result

#%%


def _generate_ER_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, weight_scale = False):

    G = nx.erdos_renyi_graph(num_nodes, graph_hyper)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 1e-02, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (1e-04) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def generate_ER_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, weight_scale):
    n_cpu = multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_ER_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result

#%%


def _generate_SBM100noise_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, weight_scale = False):

    size = [4, 2, 2, 13, 13, 15, 17, 3, 12, 10, 9]

    p = graph_hyper
    probs = [[0.95, p, p, p, p, p, p, p, p, p, p],
             [p, 1, p, p, p, p, p, p, p, p, p],
             [p, p, 1, p, p, p, p, p, p, p, p],
             [p, p, p, 0.6, p, p, p, p, p, p, p],
             [p, p, p, p, 0.6, p, p, p, p, p, p],
             [p, p, p, p, p, 0.5, p, p, p, p, p],
             [p, p, p, p, p, p, 0.5, p, p, p, p],
             [p, p, p, p, p, p, p, 0.95, p, p, p],
             [p, p, p, p, p, p, p, p, 0.65, p, p],
             [p, p, p, p, p, p, p, p, p, 0.65, p],
             [p, p, p, p, p, p, p, p, p, p, 0.65]]

    G = nx.stochastic_block_model(size, probs)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (1e-06) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def generate_SBM100noise_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, weight_scale):
    n_cpu = multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_SBM100noise_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result

def get_a_from_matrix(mat):
    new_mat = (mat != 0) * 1
    np.fill_diagonal(new_mat, 0)
    return new_mat


#%%
