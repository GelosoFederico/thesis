import pandapower
import pandapower.networks
import numpy as np
import cvxpy as cp
import matplotlib.pyplot
from NetworkMatrix import get_b_matrix_from_network, IEEE14_b_matrix
from utils import matprint, get_U_matrix
from simulations import F_score, cramer_rao_bound, MSE_matrix, get_observations
from GrotasAlgorithm import GrotasAlgorithm

two_phase_enabled = False
augmented_enabled = True

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
range_SNR = np.linspace(0, 25, 10)
for SNR in range_SNR:
    for N in [1500]:
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

        if two_phase_enabled:
            M = B.shape[0]
            U = get_U_matrix(M)
            CRB = cramer_rao_bound(M, U.T @ B @ U, sigma_est**2,sigma_p, U.T @ sigma_theta @ U, N)
            MSE_tests.append({
                "method": 'cramer_rao_bound',
                "N": N,
                "SNR": SNR,
                "MSE": CRB,
            })

if two_phase_enabled:
    MSE_two_phase = [x for x in MSE_tests if x['method']=='two_phase_topology']
    MSE_two_phase_for_plot = [x['MSE'] for x in MSE_two_phase]
    fscore_for_plot = [x['F_score'] for x in MSE_two_phase]

if augmented_enabled:
    MSE_augmented_lagrangian = [x for x in MSE_tests if x['method']=='augmented_lagrangian']

    MSE_augmented_lagrangian_for_plot = [x['MSE'] for x in MSE_augmented_lagrangian]
    fscore_augmented_lagrangian_for_plot = [x['F_score'] for x in MSE_augmented_lagrangian]

if two_phase_enabled:
    CRBs = [x for x in MSE_tests if x['method']=='cramer_rao_bound']
    CRBs_for_plot = [x['MSE'] for x in CRBs]


import matplotlib.pyplot
if two_phase_enabled:
    print("MSE_two_phase_for_plot")
    print(MSE_two_phase_for_plot)
if two_phase_enabled:
    print("CRBs_for_plot")
    print(CRBs_for_plot)
    print("fscore_for_plot")
    print(fscore_for_plot)
if augmented_enabled:
    print("MSE_augmented_lagrangian_for_plot")
    print(MSE_augmented_lagrangian_for_plot)
    print("fscore_augmented_lagrangian_for_plot")
    print(fscore_augmented_lagrangian_for_plot)
if two_phase_enabled:
    matplotlib.pyplot.plot(range_SNR, MSE_two_phase_for_plot)
    matplotlib.pyplot.plot(range_SNR, CRBs_for_plot)
if augmented_enabled:
    matplotlib.pyplot.plot(range_SNR, MSE_augmented_lagrangian_for_plot)
matplotlib.pyplot.show()
