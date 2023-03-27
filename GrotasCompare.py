import argparse
import json
import pickle
import numpy as np
from datetime import datetime

import GrotasAlgorithm
# from GrotasAlgorithm import GrotasAlgorithm
from GrotasTests import run_test
from l2g_approach.src.random_graph_rt_nested import generate_random_rt_nested_network
from utils import create_matrix_from_nx_graph

from NetworkMatrix import IEEE57_b_matrix
from simulations import get_observations
from utils import matwrite

time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

def save_as_json(run):
    B = None
    sigma_p = None
    theta = None
    if 'B' in run:
        B = run['B']
        del run['B']
    if 'theta' in run:
        theta = run['theta']
        del run['theta']
    if 'sigma_p' in run:
        sigma_p = run['sigma_p']
        del run['sigma_p']
    run['N'] = int(run['N'])
    with open(f"runs/run_grotas_compare_{time_now}.json", 'w') as fp:
        json.dump(run, fp, sort_keys=True, indent=4)
    if B is not None:
        matwrite(B, f"runs/run_grotas_compare_{time_now}_B.json")
    if sigma_p is not None:
        matwrite(sigma_p, f"runs/run_grotas_compare_{time_now}_sigma_p.json")
    if theta is not None:
        matwrite(theta, f"runs/run_grotas_compare_{time_now}_theta.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--augmented', default=False, action='store_true')
    # parser.add_argument('--two_phase', default=False, action='store_true')
    parser.add_argument('--N', default=200, type=int)
    parser.add_argument('--SNR', default=20, type=float)
    parser.add_argument('--random', default=False, action='store_true')


    parsed_args = parser.parse_args()
    # two_phase_enabled = parsed_args.two_phase
    # augmented_enabled = parsed_args.augmented
    N = parsed_args.N
    SNR = parsed_args.SNR
    
    if parsed_args.random:
        rt_nested_hyper = {
            'N': 57,
            'K':2,
            'd':4,
            'alpha':0.5,
            'beta':0.4,
            'p_rewire':0.3, 
            'N_subnetworks': 6,
            'distribution_params': (-2.4, 2.1, 2.0),
        }
        G = None
        while not G:
            try:
                G = generate_random_rt_nested_network(**rt_nested_hyper)
            except Exception as e:
                pass
        B_real, A = create_matrix_from_nx_graph(G)
    else:
        B_real, A = IEEE57_b_matrix()

    # points = np.linspace(200, 6000, 15, dtype=np.integer)
    # range_SNR = np.linspace(5, 60, 21)
    c = 1
    # parameters = [6,7,8,9,10]

    GrotasAlgorithm.augmented_lagrangian_penalty_parameter = 1e-10
    GrotasAlgorithm.augmented_lagrangian_learning_rate = 1e-10
    method = 'two_phase_topology'
    try:
        observations, sigma_theta, states, noise_sigma = get_observations(N, SNR, c, B_real)
        with open(f'data/observations_{time_now}.pickle', 'wb') as handle:
            data = {'samples': observations, 'matrix': B_real, 'matrix_a': A}
            pickle.dump(data, handle, protocol=4)
        run = run_test(B_real, observations, sigma_theta, method, states)
        run['SNR'] = SNR
        run['random'] = "true" if parsed_args.random else "false"
        print(f"++++++ Run N={N}, SNR={SNR}, method={method}\n MSE={run['MSE']}")
        print(run)
        save_as_json(run)
    except Exception as e:
        print(e)
        print(f"++++++Couldnt do it with N={N}, SNR={SNR}")


