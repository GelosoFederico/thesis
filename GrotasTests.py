import pandapower
import pandapower.networks
import json
import numpy as np
import cvxpy as cp
import matplotlib.pyplot
from datetime import datetime
from NetworkMatrix import get_b_matrix_from_network, IEEE14_b_matrix
from utils import matprint, get_U_matrix
from simulations import F_score, cramer_rao_bound, MSE_matrix, get_observations
import GrotasAlgorithm
from GrotasAlgorithm import GrotasAlgorithm
import matplotlib.pyplot

two_phase_enabled = True
augmented_enabled = True
time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

def plot_all_MSE(all_runs, N_points_arr, range_SNR):
    plots = []
    legend = []
    for N in N_points_arr:
        if two_phase_enabled:
            plots.append([x['MSE'] for x in all_runs if x['method']=='two_phase_topology' and N == x['N']])
            legend.append("MSE with two phase, N={}".format(N))
        if augmented_enabled:
            plots.append([x['MSE'] for x in all_runs if x['method']=='augmented_lagrangian' and N == x['N']])
            legend.append("MSE with augmented Lagrangian, N={}".format(N))
        plots.append([x['MSE'] for x in all_runs if x['method']=='cramer_rao_bound' and N == x['N']])
        legend.append("CBR, N={}".format(N))
    

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    colors = ['red', 'blue', 'black', 'magenta', 'green', 'black']
    color_gen = (x for x in colors)
    for plot in plots:
        ax.semilogy(range_SNR, plot,  color=next(color_gen), lw=1)
    ax.set_title("MSE for 14-bus network")
    ax.set_yscale('log')
    fig.legend(legend)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('MSE')
    matplotlib.pyplot.xlabel('SNR [dB]')
    matplotlib.pyplot.savefig('plots/MSE_14_bus{}.png'.format(time_now))
    matplotlib.pyplot.show()

def plot_B_matrix(all_runs, N_points_arr, B_real):
    target_SNR = 15

    for N in N_points_arr:
        if two_phase_enabled:
            B = [x for x in all_runs if x['method']=='two_phase_topology' and N == x['N'] and x["SNR"] == target_SNR]
            B = B[0]
            snr = B['SNR']
            B_matrix = B['B']
            matplotlib.pyplot.matshow(B_matrix)
            matplotlib.pyplot.title("B matrix, N={}, SNR={} dB".format(str(N), str(snr)))
            matplotlib.pyplot.savefig('plots/two_phase_topology_B_matrix{}.png'.format(time_now))
            matplotlib.pyplot.show()
        if augmented_enabled:
            B = [x for x in all_runs if x['method']=='augmented_lagrangian' and N == x['N'] and x["SNR"] == target_SNR]
            B = B[0]
            snr = B['SNR']
            B_matrix = B['B']
            matplotlib.pyplot.matshow(B_matrix)
            matplotlib.pyplot.title("B matrix, N={}, SNR={} dB".format(str(N), str(snr)))
            matplotlib.pyplot.savefig('plots/augmented_lagrangian_B_matrix{}.png'.format(time_now))
            matplotlib.pyplot.show()
    matplotlib.pyplot.matshow(B_real)
    matplotlib.pyplot.title("real B matrix".format(str(N), str(snr)))
    matplotlib.pyplot.savefig('plots/real_B_matrix{}.png'.format(time_now))
    matplotlib.pyplot.show()

def basic_plot_checks(all_runs, N_points_arr, range_SNR):
    for N in N_points_arr:
        if two_phase_enabled:
            MSE_two_phase = [x for x in all_runs if x['method']=='two_phase_topology' and N == x['N']]
            MSE_two_phase_for_plot = [x['MSE'] for x in MSE_two_phase if N == x['N']]

        if augmented_enabled:
            MSE_augmented_lagrangian = [x for x in all_runs if x['method']=='augmented_lagrangian' and N == x['N']]

            MSE_augmented_lagrangian_for_plot = [x['MSE'] for x in MSE_augmented_lagrangian if N == x['N']]

        CRBs = [x for x in all_runs if x['method']=='cramer_rao_bound' and N == x['N']]
        CRBs_for_plot = [x['MSE'] for x in CRBs if  N == x['N']]
        if two_phase_enabled:
            matplotlib.pyplot.plot(range_SNR, MSE_two_phase_for_plot)
        if augmented_enabled:
            matplotlib.pyplot.plot(range_SNR, MSE_augmented_lagrangian_for_plot)
        matplotlib.pyplot.plot(range_SNR, CRBs_for_plot)
        matplotlib.pyplot.title("MSE for {}".format(str(N)))
        matplotlib.pyplot.grid()
        matplotlib.pyplot.xlabel('MSE')
        matplotlib.pyplot.ylabel('SNR [dB]')
        matplotlib.pyplot.show()
       
def basic_plot_prints(all_runs, N_points_arr):
    for N in N_points_arr:
        if two_phase_enabled:
            MSE_two_phase = [x for x in all_runs if x['method']=='two_phase_topology' and N == x['N']]
            MSE_two_phase_for_plot = [x['MSE'] for x in MSE_two_phase if N == x['N']]
            fscore_for_plot = [x['F_score'] for x in MSE_two_phase if N == x['N']]

        if augmented_enabled:
            MSE_augmented_lagrangian = [x for x in all_runs if x['method']=='augmented_lagrangian' and N == x['N']]
            MSE_augmented_lagrangian_for_plot = [x['MSE'] for x in MSE_augmented_lagrangian if N == x['N']]
            fscore_augmented_lagrangian_for_plot = [x['F_score'] for x in MSE_augmented_lagrangian if N == x['N']]

        CRBs = [x for x in all_runs if x['method']=='cramer_rao_bound' and N == x['N']]
        CRBs_for_plot = [x['MSE'] for x in CRBs if  N == x['N']]
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

def plot_all_fscore(all_runs, N_points_arr, B_real):
    plots = []
    legend = []
    for N in N_points_arr:
        if two_phase_enabled:
            plots.append([x['F_score'] for x in all_runs if x['method']=='two_phase_topology' and N == x['N']])
            legend.append("F_score with two phase, N={}".format(N))
        if augmented_enabled:
            plots.append([x['F_score'] for x in all_runs if x['method']=='augmented_lagrangian' and N == x['N']])
            legend.append("F_score with augmented Lagrangian, N={}".format(N))
    

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    colors = ['red', 'blue', 'black', 'magenta', 'green', 'black']
    color_gen = (x for x in colors)
    for plot in plots:
        ax.semilogy(range_SNR, plot,  color=next(color_gen), lw=1)
    ax.set_title("F_score for 14-bus network")
    fig.legend(legend)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.xlabel('F-score')
    matplotlib.pyplot.ylabel('SNR [dB]')
    matplotlib.pyplot.savefig('plots/f_score_14_bus{}.png'.format(time_now))
    matplotlib.pyplot.show()


def run_test(B_real, observations, sigma_theta, method):
    B, theta, sigma_est, sigma_p = GrotasAlgorithm(observations, sigma_theta, method)
    MSE = MSE_matrix(B_real, B)
    fs = F_score(B, B_real)
    print(f"{fs=}")
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
        "F_score": fs
    }

def run_cramer_rao_bound(B, sigma_est, sigma_p, sigma_theta, N):
    M = B.shape[0]
    U = get_U_matrix(M)
    CRB = cramer_rao_bound(M, U.T @ B @ U, sigma_est**2,sigma_p, U.T @ sigma_theta @ U, N)
    return{
        "method": 'cramer_rao_bound',
        "N": N,
        "MSE": CRB,
    }

def save_as_jason(all_runs):
    serializable_things = []
    for run in all_runs:
        new_run = run.copy()
        if 'B' in new_run:
            del new_run['B']
        serializable_things.append(new_run)
        if 'theta' in new_run:
            del new_run['theta']
        if 'sigma_p' in new_run:
            del new_run['sigma_p']
    with open("runs/run_{}.json".format(time_now), 'w') as fp:
        json.dump(serializable_things, fp, sort_keys=True, indent=4)

if __name__ == '__main__':

    net = pandapower.networks.case14()
    pandapower.runpp(net)
    MSE_tests = []
    B_real, A = get_b_matrix_from_network(net)
    B_real, A = IEEE14_b_matrix()
    c = 1
    range_SNR = np.linspace(0, 25, 21)
    # range_SNR = [0]
    points = [200, 1500]
    # GrotasAlgorithm.augmented_lagrangian_penalty_parameter = 0.2
    # GrotasAlgorithm.augmented_lagrangian_learning_rate = 0.1

    MSE_tests = []
    for SNR in range_SNR:
        for N in points:
            sigma_est = None
            sigma_p = None
            observations, sigma_theta = get_observations(N, SNR, c, B_real)
            if augmented_enabled:
                run = run_test(B_real, observations, sigma_theta, 'augmented_lagrangian')
                run['SNR'] = SNR
                MSE_tests.append(run)
                sigma_est = run['sigma_est']
                sigma_p = run['sigma_p']
            if two_phase_enabled:
                run = run_test(B_real, observations, sigma_theta, 'two_phase_topology')
                run['SNR'] = SNR
                MSE_tests.append(run)
                sigma_est = run['sigma_est']
                sigma_p = run['sigma_p']

            run = run_cramer_rao_bound(B_real, sigma_est, sigma_p, sigma_theta, N)
            run['SNR'] = SNR
            MSE_tests.append(run)

    basic_plot_prints(MSE_tests, points)
    basic_plot_checks(MSE_tests, points, range_SNR)

    # Now we do every plot in Grotas's paper
    plot_B_matrix(MSE_tests, points, B_real)
    plot_all_MSE(MSE_tests, points, range_SNR)
    plot_all_fscore(MSE_tests, points, range_SNR)
    save_as_jason(MSE_tests)


