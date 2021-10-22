import GrotasAlgorithm
import GrotasTests
from NetworkMatrix import IEEE14_b_matrix, IEEE118_b_matrix
import numpy as np

from utils import matprint


B_real, A = IEEE118_b_matrix()
SNR = 10
points = 300

nabla_values = np.geomspace(1e-12, 1e-1, 12)
gamma_values = np.geomspace(1e-12, 1e-1, 12)

matrix_values = np.zeros((nabla_values.shape[0], gamma_values.shape[0]))

observations, sigma_theta, states, noise_sigma = GrotasTests.get_observations(points, SNR, 1, B_real)

for nabla in range(0, len(nabla_values)):
    GrotasAlgorithm.augmented_lagrangian_learning_rate = nabla_values[nabla]
    for gamma in range(0, len(gamma_values)):
        GrotasAlgorithm.augmented_lagrangian_penalty_parameter = gamma_values[gamma]
        try:
            result = GrotasTests.run_test(B_real, observations, sigma_theta, 'augmented_lagrangian', states)
            matrix_values[nabla][gamma] = result['MSE']
        except Exception as e:
            print(e)
            print(e.args)
            matrix_values[nabla][gamma] = e.args[0]
        print("Just run nabla pos {}, gamma pos {}".format(str(nabla), str(gamma)))

print(nabla_values)
print(gamma_values)
matprint(matrix_values)

