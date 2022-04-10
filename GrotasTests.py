import argparse
import json
from datetime import datetime
from typing import List

import matplotlib.pyplot
import numpy as np
import pandapower
import pandapower.networks

import GrotasAlgorithm
from GrotasAlgorithm import GrotasAlgorithm
from NetworkMatrix import (IEEE14_b_matrix, IEEE118_b_matrix,
                           get_b_matrix_from_network)
from simulations import (F_score, MSE_matrix, MSE_states, MSE_states_theoretical, cramer_rao_bound,
                         get_observations)
from utils import get_U_matrix, matprint, matwrite

two_phase_enabled = True
augmented_enabled = False
ieee14 = False
ieee118 = False
time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")


def plot_all_MSE(all_runs, N_points_arr, range_SNR):
    plots = []
    legend = []
    for N in N_points_arr:
        if two_phase_enabled:
            plots.append([x['MSE'] for x in all_runs if x['method'] == 'two_phase_topology' and N == x['N']])
            legend.append("MSE with two phase, N={}".format(N))
        if augmented_enabled:
            plots.append([x['MSE'] for x in all_runs if x['method'] == 'augmented_lagrangian' and N == x['N']])
            legend.append("MSE with augmented Lagrangian, N={}".format(N))
        plots.append([x['MSE'] for x in all_runs if x['method'] == 'cramer_rao_bound' and N == x['N']])
        legend.append("CBR, N={}".format(N))

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['red', 'blue', 'black', 'magenta', 'green', 'grey']
    color_gen = (x for x in colors)
    for plot in plots:
        ax.semilogy(range_SNR, plot, color=next(color_gen), lw=1)
    ax.set_title("MSE for 14-bus network")
    ax.set_yscale('log')
    fig.legend(legend)
    matplotlib.pyplot.grid(True, which='both')
    matplotlib.pyplot.ylabel('MSE')
    matplotlib.pyplot.xlabel('SNR [dB]')
    matplotlib.pyplot.savefig('plots/MSE_14_bus{}.png'.format(time_now))
    matplotlib.pyplot.show()


def plot_all_MSE_by_points(all_runs, N_points_arr, range_SNR):
    plots = []
    legend = []
    for SNR in range_SNR:
        if two_phase_enabled:
            plots.append([x['MSE'] for x in all_runs if x['method'] == 'two_phase_topology' and SNR == x['SNR']])
            legend.append("MSE with two phase, SNR={}".format(SNR))
        if augmented_enabled:
            plots.append([x['MSE'] for x in all_runs if x['method'] == 'augmented_lagrangian' and SNR == x['SNR']])
            legend.append("MSE with augmented Lagrangian, SNR={}".format(SNR))
        plots.append([x['MSE'] for x in all_runs if x['method'] == 'cramer_rao_bound' and SNR == x['SNR']])
        legend.append("CBR, SNR={}".format(SNR))

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['red', 'blue', 'black', 'magenta', 'green', 'black']
    color_gen = (x for x in colors)
    for plot in plots:
        ax.semilogy(N_points_arr, plot, color=next(color_gen), lw=1)
    ax.set_title("MSE for 118-bus network")
    ax.set_yscale('log')
    fig.legend(legend)
    matplotlib.pyplot.grid(True, which='both')
    matplotlib.pyplot.ylabel('MSE')
    matplotlib.pyplot.xlabel('points')
    matplotlib.pyplot.savefig('plots/MSE_118_bus{}.png'.format(time_now))
    matplotlib.pyplot.show()


def plot_all_MSE_states_by_points(all_runs, N_points_arr, range_SNR):
    plots = []
    legend = []
    for SNR in range_SNR:
        if two_phase_enabled:
            plots.append([x['MSE_states'] for x in all_runs if x['method'] == 'two_phase_topology' and SNR == x['SNR']])
            legend.append("MSE with two phase, SNR={}".format(SNR))
        if augmented_enabled:
            plots.append([x['MSE_states'] for x in all_runs if x['method'] == 'augmented_lagrangian' and SNR == x['SNR']])
            legend.append("MSE with augmented Lagrangian, SNR={}".format(SNR))
        plots.append([x['MSE_states'] for x in all_runs if x['method'] == 'MSE_oracle' and SNR == x['SNR']])
        legend.append("oracle, SNR={}".format(SNR))

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['red', 'blue', 'black', 'magenta', 'green', 'black']
    color_gen = (x for x in colors)
    for plot in plots:
        ax.semilogy(N_points_arr, plot, color=next(color_gen), lw=1)
    ax.set_title("MSE states for 118-bus network")
    ax.set_yscale('log')
    fig.legend(legend)
    matplotlib.pyplot.grid(True, which='both')
    matplotlib.pyplot.ylabel('MSE')
    matplotlib.pyplot.xlabel('Points')
    matplotlib.pyplot.savefig('plots/MSE_states_118_bus{}.png'.format(time_now))
    matplotlib.pyplot.show()


def plot_all_fscore_by_points(all_runs, N_points, range_SNR):
    plots = []
    legend = []
    for SNR in range_SNR:
        if two_phase_enabled:
            plots.append([x['F_score'] for x in all_runs if x['method'] == 'two_phase_topology' and SNR == x['SNR']])
            legend.append("F_score with two phase, SNR={}".format(SNR))
        if augmented_enabled:
            plots.append([x['F_score'] for x in all_runs if x['method'] == 'augmented_lagrangian' and SNR == x['SNR']])
            legend.append("F_score with augmented Lagrangian, SNR={}".format(SNR))
    
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['red', 'blue', 'black', 'magenta', 'green', 'black']
    color_gen = (x for x in colors)
    for plot in plots:
        ax.semilogy(N_points, plot, color=next(color_gen), lw=1)
    ax.set_title("F_score for 118-bus network")
    fig.legend(legend)
    matplotlib.pyplot.grid(True, which='both')
    matplotlib.pyplot.xlabel('points')
    matplotlib.pyplot.ylabel('F-score')
    matplotlib.pyplot.savefig('plots/f_score_118_bus{}.png'.format(time_now))
    matplotlib.pyplot.show()

def plot_all_MSE_states(all_runs, N_points_arr, range_SNR):
    plots = []
    legend = []
    for N in N_points_arr:
        if two_phase_enabled:
            plots.append([x['MSE_states'] for x in all_runs if x['method'] == 'two_phase_topology' and N == x['N']])
            legend.append("MSE with two phase, N={}".format(N))
        if augmented_enabled:
            plots.append([x['MSE_states'] for x in all_runs if x['method'] == 'augmented_lagrangian' and N == x['N']])
            legend.append("MSE with augmented Lagrangian, N={}".format(N))
        plots.append([x['MSE_states'] for x in all_runs if x['method'] == 'MSE_oracle' and N == x['N']])
        legend.append("oracle, N={}".format(N))
    plots.append([x['MSE_states'] for x in all_runs if x['method'] == 'MSE_theory'])
    legend.append("theory".format(N))

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['red', 'blue', 'black', 'magenta', 'green', 'grey', 'yellow']
    color_gen = (x for x in colors)
    for plot in plots:
        ax.semilogy(range_SNR, plot, color=next(color_gen), lw=1)
    ax.set_title("MSE states for 14-bus network")
    ax.set_yscale('log')
    fig.legend(legend)
    matplotlib.pyplot.grid(True, which='both')
    matplotlib.pyplot.ylabel('MSE')
    matplotlib.pyplot.xlabel('SNR [dB]')
    matplotlib.pyplot.savefig('plots/MSE_states_14_bus{}.png'.format(time_now))
    matplotlib.pyplot.show()


def plot_B_matrix(all_runs, N_points_arr, B_real):
    target_SNR = 45 if ieee118 else 15 

    for N in N_points_arr:
        if two_phase_enabled:
            B = [x for x in all_runs if x['method'] == 'two_phase_topology' and N == x['N'] and x["SNR"] == target_SNR]
            B = B[0]
            snr = B['SNR']
            B_matrix = B['B']
            matplotlib.pyplot.matshow(B_matrix)
            matplotlib.pyplot.title("B matrix, N={}, SNR={} dB".format(str(N), str(snr)))
            matplotlib.pyplot.savefig('plots/two_phase_topology_B_matrix{}.png'.format(time_now))
            matplotlib.pyplot.show()
        if augmented_enabled:
            B = [x for x in all_runs if x['method'] == 'augmented_lagrangian' and N == x['N'] and x["SNR"] == target_SNR]
            B = B[0]
            snr = B['SNR']
            B_matrix = B['B']
            matplotlib.pyplot.matshow(B_matrix)
            matplotlib.pyplot.title("B matrix, N={}, SNR={} dB".format(str(N), str(snr)))
            matplotlib.pyplot.savefig('plots/augmented_lagrangian_B_matrix{}.png'.format(time_now))
            matplotlib.pyplot.show()
    matplotlib.pyplot.matshow(B_real)
    matplotlib.pyplot.title("real B matrix")
    matplotlib.pyplot.savefig('plots/real_B_matrix{}.png'.format(time_now))
    matplotlib.pyplot.show()


def basic_plot_checks(all_runs, N_points_arr, range_SNR):
    for N in N_points_arr:
        if two_phase_enabled:
            MSE_two_phase = [x for x in all_runs if x['method'] == 'two_phase_topology' and N == x['N']]
            MSE_two_phase_for_plot = [x['MSE'] for x in MSE_two_phase if N == x['N']]

        if augmented_enabled:
            MSE_augmented_lagrangian = [x for x in all_runs if x['method'] == 'augmented_lagrangian' and N == x['N']]

            MSE_augmented_lagrangian_for_plot = [x['MSE'] for x in MSE_augmented_lagrangian if N == x['N']]

        CRBs = [x for x in all_runs if x['method'] == 'cramer_rao_bound' and N == x['N']]
        CRBs_for_plot = [x['MSE'] for x in CRBs if N == x['N']]
        if two_phase_enabled:
            matplotlib.pyplot.plot(range_SNR, MSE_two_phase_for_plot)
        if augmented_enabled:
            matplotlib.pyplot.plot(range_SNR, MSE_augmented_lagrangian_for_plot)
        matplotlib.pyplot.plot(range_SNR, CRBs_for_plot)
        matplotlib.pyplot.title("MSE for {}".format(str(N)))
        matplotlib.pyplot.grid(True, which='both')
        matplotlib.pyplot.ylabel('MSE')
        matplotlib.pyplot.xlabel('SNR [dB]')
        matplotlib.pyplot.show()


def basic_plot_prints(all_runs, N_points_arr):
    for N in N_points_arr:
        if two_phase_enabled:
            MSE_two_phase = [x for x in all_runs if x['method'] == 'two_phase_topology' and N == x['N']]
            MSE_two_phase_for_plot = [x['MSE'] for x in MSE_two_phase if N == x['N']]
            fscore_for_plot = [x['F_score'] for x in MSE_two_phase if N == x['N']]

        if augmented_enabled:
            MSE_augmented_lagrangian = [x for x in all_runs if x['method'] == 'augmented_lagrangian' and N == x['N']]
            MSE_augmented_lagrangian_for_plot = [x['MSE'] for x in MSE_augmented_lagrangian if N == x['N']]
            fscore_augmented_lagrangian_for_plot = [x['F_score'] for x in MSE_augmented_lagrangian if N == x['N']]

        CRBs = [x for x in all_runs if x['method'] == 'cramer_rao_bound' and N == x['N']]
        CRBs_for_plot = [x['MSE'] for x in CRBs if N == x['N']]
    if two_phase_enabled:
        print("MSE_two_phase_for_plot")
        print(MSE_two_phase_for_plot)
        print("fscore_for_plot")
        print(fscore_for_plot)
    if augmented_enabled:
        print("MSE_augmented_lagrangian_for_plot")
        print(MSE_augmented_lagrangian_for_plot)
        print("fscore_augmented_lagrangian_for_plot")
        print(fscore_augmented_lagrangian_for_plot)
    print("CRBs_for_plot")
    print(CRBs_for_plot)


def plot_all_fscore(all_runs, N_points_arr, range_SNR):
    plots = []
    legend = []
    for N in N_points_arr:
        if two_phase_enabled:
            plots.append([x['F_score'] for x in all_runs if x['method'] == 'two_phase_topology' and N == x['N']])
            legend.append("F_score with two phase, N={}".format(N))
        if augmented_enabled:
            plots.append([x['F_score'] for x in all_runs if x['method'] == 'augmented_lagrangian' and N == x['N']])
            legend.append("F_score with augmented Lagrangian, N={}".format(N))
    
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['red', 'blue', 'black', 'magenta', 'green', 'black']
    color_gen = (x for x in colors)
    for plot in plots:
        ax.semilogy(range_SNR, plot, color=next(color_gen), lw=1)
    ax.set_title("F_score for 14-bus network")
    fig.legend(legend)
    matplotlib.pyplot.grid(True, which='both')
    matplotlib.pyplot.xlabel('SNR [dB]')
    matplotlib.pyplot.ylabel('F-score')
    matplotlib.pyplot.savefig('plots/f_score_14_bus{}.png'.format(time_now))
    matplotlib.pyplot.show()


def run_test(B_real, observations, sigma_theta, method, states):
    B, theta, sigma_est, sigma_p = GrotasAlgorithm(observations, sigma_theta, method)
    MSE = MSE_matrix(B_real, B)
    fs = F_score(B, B_real)
    MSE_states_total = MSE_states(observations, B, sigma_theta, sigma_est**2, states)
    # print(f"{fs=}")
    # print("B_found")
    # matprint(B)
    return {
        "method": method,
        "N": observations.shape[0],
        "B": B,
        "theta": theta,
        "sigma_est": sigma_est,
        "sigma_p": sigma_p,
        "MSE": MSE,
        "MSE_states": MSE_states_total,
        "F_score": fs
    }


def run_cramer_rao_bound(B, sigma_est, sigma_p, sigma_theta, N):
    M = B.shape[0]
    U = get_U_matrix(M)
    CRB = cramer_rao_bound(M, U.T @ B @ U, sigma_est**2, sigma_p, U.T @ sigma_theta @ U, N)
    return{
        "method": 'cramer_rao_bound',
        "N": N,
        "MSE": CRB,
    }


def run_MSE_states_oracle(observations, B, sigma_theta, sigma_est, states):
    N = observations.shape[0]
    MSE_states_total = MSE_states(observations, B, sigma_theta, sigma_est, states)
    # matprint(MSE_states_total)
    return{
        "method": 'MSE_oracle',
        "N": N,
        "MSE_states": MSE_states_total,
    }


def run_MSE_states_theory(B, sigma_theta, sigma_noise):
    # B should be real b
    M = B.shape[0]
    MSE_states_total = np.trace(sigma_theta - sigma_theta @ B.T @ np.linalg.inv(B @ sigma_theta @ B.T + sigma_noise * np.eye(M)) @ B @ sigma_theta)
    return{
        "method": 'MSE_theory',
        "MSE_states": MSE_states_total
    }


def save_as_json(all_runs):
    serializable_things = []
    ultimate_b = None
    for run in all_runs:
        new_run = run.copy()
        if 'B' in new_run:
            ultimate_b = new_run['B']
            del new_run['B']
        serializable_things.append(new_run)
        if 'theta' in new_run:
            del new_run['theta']
        if 'sigma_p' in new_run:
            del new_run['sigma_p']
        new_run['N'] = int(new_run['N'])
    with open("runs/run_{}_{}.json".format(time_now, 'IEEE118' if ieee118 else 'IEEE14'), 'w') as fp:
        json.dump(serializable_things, fp, sort_keys=True, indent=4)
    if ieee118:
        matwrite(ultimate_b, "runs/runIEEE118_{}.json".format(time_now))


def smooth_MSE_tests(MSE_tests: List[dict]) -> List:
    new_MSE = []
    tuples_found = set()
    for item in MSE_tests:
        defining_tuple = (item['SNR'], item['N'], item['method'])
        if defining_tuple not in tuples_found:
            similar_MSE = [x for x in MSE_tests if x['SNR'] == item['SNR'] and x['N'] == item['N'] and x['method'] == item['method']]
            averaged_MSE = {
                'SNR': item['SNR'],
                'N': item['N'],
                'method': item['method'],
            }
            for key in similar_MSE[0].keys():
                if key not in ['SNR', 'N', 'method']:
                    averaged_MSE[key] = np.mean([x[key] for x in similar_MSE], axis=0)
            tuples_found.add(defining_tuple)
            new_MSE.append(averaged_MSE)
    return new_MSE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ieee14', default=False, action='store_true')
    parser.add_argument('--ieee118', default=False, action='store_true')
    parser.add_argument('--augmented', default=False, action='store_true')
    parser.add_argument('--two_phase', default=False, action='store_true')
    parser.add_argument('--smoothing_points', default=1, type=int)
    parsed_args = parser.parse_args()
    ieee14 = parsed_args.ieee14
    ieee118 = parsed_args.ieee118
    two_phase_enabled = parsed_args.two_phase
    augmented_enabled = parsed_args.augmented
    smoothing_points = parsed_args.smoothing_points

    MSE_tests = []
    if ieee118:
        B_real, A = IEEE118_b_matrix()
        range_SNR = [45]
        points = np.linspace(800, 1000, 2, dtype=np.integer)
        c = np.sqrt(10)
        GrotasAlgorithm.augmented_lagrangian_penalty_parameter = 1e-10
        GrotasAlgorithm.augmented_lagrangian_learning_rate = 1e-10
    else:
        # net = pandapower.networks.case14()
        # pandapower.runpp(net)
        # B_real, A = get_b_matrix_from_network(net)
        B_real, A = IEEE14_b_matrix()
        c = 1
        range_SNR = np.linspace(5, 25, 21)
        points = [200, 1500]
        GrotasAlgorithm.augmented_lagrangian_penalty_parameter = 1e-7
        GrotasAlgorithm.augmented_lagrangian_learning_rate = 1e-7

    MSE_tests = []
    # smoothing_points = 10
    for full_run in range(smoothing_points):
        for SNR in range_SNR:
            for N in points:
                sigma_est = None
                sigma_p = None
                observations, sigma_theta, states, noise_sigma = get_observations(N, SNR, c, B_real)
                if augmented_enabled:
                    run = run_test(B_real, observations, sigma_theta, 'augmented_lagrangian', states)
                    run['SNR'] = SNR
                    MSE_tests.append(run)
                    sigma_est = run['sigma_est']
                    sigma_p = run['sigma_p']
                if two_phase_enabled:
                    run = run_test(B_real, observations, sigma_theta, 'two_phase_topology', states)
                    run['SNR'] = SNR
                    MSE_tests.append(run)
                    sigma_est = run['sigma_est']
                    sigma_p = run['sigma_p']

                run = run_cramer_rao_bound(B_real, sigma_est, sigma_p, sigma_theta, N)
                run['SNR'] = SNR
                MSE_tests.append(run)

                run = run_MSE_states_oracle(observations, B_real, sigma_theta, noise_sigma, states)
                run['SNR'] = SNR
                MSE_tests.append(run)

            run = run_MSE_states_theory(B_real, sigma_theta, noise_sigma)
            run['SNR'] = SNR
            run['N'] = points[0]
            MSE_tests.append(run)

    smoothed_MSE_tests = smooth_MSE_tests(MSE_tests)
    basic_plot_prints(smoothed_MSE_tests, points)

    # Now we do every plot in Grotas's paper
    if ieee118:
        plot_all_MSE_by_points(smoothed_MSE_tests, points, range_SNR)
        plot_all_MSE_states_by_points(smoothed_MSE_tests, points, range_SNR)
        plot_all_fscore_by_points(smoothed_MSE_tests, points, range_SNR)
        plot_B_matrix(smoothed_MSE_tests, points, B_real)
        save_as_json(smoothed_MSE_tests)
    else:
        basic_plot_checks(smoothed_MSE_tests, points, range_SNR)
        plot_B_matrix(smoothed_MSE_tests, points, B_real)
        plot_all_MSE(smoothed_MSE_tests, points, range_SNR)
        plot_all_MSE_states(smoothed_MSE_tests, points, range_SNR)
        plot_all_fscore(smoothed_MSE_tests, points, range_SNR)
        save_as_json(smoothed_MSE_tests)
