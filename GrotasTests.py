import pandapower
import pandapower.networks
import numpy as np
import cvxpy as cp
import matplotlib.pyplot
from datetime import datetime
from NetworkMatrix import get_b_matrix_from_network, IEEE14_b_matrix
from utils import matprint, get_U_matrix
from simulations import F_score, cramer_rao_bound, MSE_matrix, get_observations
from GrotasAlgorithm import GrotasAlgorithm
import matplotlib.pyplot

two_phase_enabled = True
augmented_enabled = False
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
    matplotlib.pyplot.xlabel('MSE')
    matplotlib.pyplot.ylabel('SNR [dB]')
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
            B = [x for x in all_runs if x['method']=='' and N == x['N'] and x["SNR"] == target_SNR]
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

def basic_plot_checks(all_runs, N_points_arr):
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


    


net = pandapower.networks.case14()
pandapower.runpp(net)
MSE_tests = []
B_real, A = get_b_matrix_from_network(net)
B_real, A = IEEE14_b_matrix()
print("B_Real_tilde")
# U = get_U_matrix(B_real.shape[0])
B_real_tilde = B_real[1:,1:]
matprint(B_real_tilde)
c = 1
range_SNR = np.linspace(0, 25, 21)
print(range_SNR)
points = [200, 1500]
for SNR in range_SNR:
    for N in points:
        sigma_est = None
        sigma_est_aug = None
        sigma_p = None
        sigma_p_aug = None
        observations, sigma_theta = get_observations(N, SNR, c, B_real)
        
        if two_phase_enabled:
            B, theta, sigma_est, sigma_p = GrotasAlgorithm(observations, sigma_theta, 'two_phase_topology')
            MSE = MSE_matrix(B_real, B)
            fs = F_score(B, B_real)
            print(f"{fs=}")
            print("B_found")
            matprint(B)
            MSE_tests.append({
                "method": 'two_phase_topology',
                "N": N,
                "SNR": SNR,
                "B": B,
                "theta": theta,
                "sigma_est": sigma_est,
                "sigma_p": sigma_p,
                "MSE": MSE,
                "F_score": fs
            })
        if augmented_enabled:
            B_aug, theta_aug, sigma_est_aug, sigma_p_aug = GrotasAlgorithm(observations, sigma_theta, 'augmented_lagrangian')
            MSE_aug = MSE_matrix(B_real, B_aug)
            fs_aug = F_score(B_aug, B_real)
            print(f"{fs_aug=}")
            print("B_found")
            matprint(B_aug)
            MSE_tests.append({
                "method": 'augmented_lagrangian',
                "N": N,
                "SNR": SNR,
                "B": B_aug,
                "theta": theta_aug,
                "sigma_est": sigma_est_aug,
                "sigma_p": sigma_p_aug,
                "MSE": MSE_aug,
                "F_score": fs_aug
            })

        M = B.shape[0]
        U = get_U_matrix(M)
        sigma_crb = sigma_est if sigma_est.all() else sigma_est_aug
        sigma_p_crb = sigma_p if sigma_p.all() else sigma_p_aug
        CRB = cramer_rao_bound(M, U.T @ B @ U, sigma_crb**2,sigma_p, U.T @ sigma_theta @ U, N)
        MSE_tests.append({
            "method": 'cramer_rao_bound',
            "N": N,
            "SNR": SNR,
            "MSE": CRB,
        })

basic_plot_prints(MSE_tests, points)
basic_plot_checks(MSE_tests, points)

# Now we do every plot in Grotas's paper
plot_B_matrix(MSE_tests, points, B_real)
plot_all_MSE(MSE_tests, points, range_SNR)
plot_all_fscore(MSE_tests, points, range_SNR)


