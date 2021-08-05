import pandapower
import pandapower.networks
import numpy as np
import scipy.linalg
import cvxpy as cp
import matplotlib.pyplot
from NetworkMatrix import get_b_matrix_from_network, IEEE14_b_matrix
from utils import matprint, get_U_matrix
from simulations import F_score, cramer_rao_bound, MSE_matrix
from GrotasAlgorithm import GrotasAlgorithm

def get_observations(N, SNR, c, B):
    M = B.shape[0]
    # sigma_theta = c**2*I_M
    # SNR = 10 log(1/sigma**2 * Tr{B_tilde * sigma_theta_tilde * B_tilde})
    # sigma ** 2 = Tr{B_tilde * sigma_theta_tilde * B_tilde} / 10 ^ (SNR/10)
    U = get_U_matrix(M)
    sigma_theta = c**2 * np.eye(M)
    sigma_theta_tilde = U.T @ sigma_theta @ U
    B_tilde = B[1:M, 1:M]
    
    noise_sigma = np.trace(B_tilde @ sigma_theta_tilde @ B_tilde) / (10**(SNR/10))

    # TODO send distribution as param
    theta_created = np.random.default_rng().normal(0, np.sqrt(c**2), (M,N))
    noise =  np.random.default_rng().normal(0, noise_sigma, (M,N))
    return ((B @ theta_created) + noise).T, sigma_theta

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
        
        B, theta, sigma_est, sigma_p = GrotasAlgorithm(observations, sigma_theta, 'two_phase_topology')
        # B, theta, sigma_est, sigma_p = GrotasAlgorithm(observations, sigma_theta, 'augmented_lagrangian')
        MSE = MSE_matrix(B_real, B)
        fs = F_score(B_real, B)
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

        M = B.shape[0]
        U = get_U_matrix(M)
        CRB = cramer_rao_bound(M, U.T @ B @ U, sigma_est**2,sigma_p, U.T @ sigma_theta @ U, N)
        MSE_tests.append({
            "method": 'cramer_rao_bound',
            "N": N,
            "SNR": SNR,
            "MSE": CRB,
        })

MSE_two_phase = [x for x in MSE_tests if x['method']=='two_phase_topology']

MSE_two_phase_for_plot = [x['MSE'] for x in MSE_two_phase]
fscore_for_plot = [x['F_score'] for x in MSE_two_phase]

CRBs = [x for x in MSE_tests if x['method']=='cramer_rao_bound']
CRBs_for_plot = [x['MSE'] for x in CRBs]


import matplotlib.pyplot
print("MSE_two_phase_for_plot")
print(MSE_two_phase_for_plot)
print("CRBs_for_plot")
print(CRBs_for_plot)
print("fscore_for_plot")
print(fscore_for_plot)
matplotlib.pyplot.plot(range_SNR, MSE_two_phase_for_plot)
matplotlib.pyplot.plot(range_SNR, CRBs_for_plot)
matplotlib.pyplot.show()
