import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from GrotasAlgorithm import GrotasAlgorithm
from NetworkMatrix import IEEE57_b_matrix
from l2g_approach.src.utils_data import get_a_from_matrix, get_z_and_w_gt, rotate_matrix
from scipy.spatial.distance import squareform
from l2g_approach.src.models import load_l2g_from_disk

def main():
    # time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_to_use_date = "20230215014040"
    net = load_l2g_from_disk(f"l2g_approach\\saved_model\\net_{model_to_use_date}").to(device)


    batch_size = 32

    # Use the IEEE 57 options
    num_samples = batch_size
    matrixes = []
    for i in range(num_samples):
        new_ieee57, a = IEEE57_b_matrix()
        if i != 0:
            new_ieee57 = rotate_matrix(new_ieee57)
            a = get_a_from_matrix(new_ieee57)
            matrixes.append((new_ieee57, a))

    final_w = []
    all_z = []
    all_samples = []
    all_states = []

    for mat, a in matrixes:
        z, W_GT, samples, states = get_z_and_w_gt(mat, mat.shape[0], 300, a)
        all_z.append(z)
        final_w.append(W_GT)
        all_samples.append(samples)
        all_states.append(states)
    
    w = []
    for mat in matrixes:
        # other_mat = mat[0].copy()
        # np.fill_diagonal(other_mat, 0)
        w.append(mat[0])
    data = TensorDataset(torch.Tensor(all_z), torch.Tensor(w))
    test_loader = DataLoader(data, batch_size=batch_size)

    num_unroll = net.layers

    for z, w_gt_batch in test_loader:
        z = z.to(device)

        w_list = net.validation(z, threshold=1e-2)
        w_pred = torch.clamp(w_list[:, num_unroll - 1, :], min=0)

    # we get the matrix like w_pred[idx,:].detach().cpu().numpy()
    pred = squareform(w_pred[0,:].detach().cpu().numpy())

    for i in range(len(all_samples)):
        B, theta, sigma_est, sigma_p = GrotasAlgorithm(all_samples[i].T, np.eye(pred.shape[0]), 'dsa')
        if i > 0:
            break
    plt.figure()
    plt.matshow(pred)
    plt.title('prediction l2g')
    # plt.savefig(f'plots_load/prediction_{idx}_model{model_to_use_date}_{time_now}.png')
    plt.show()

    plt.figure()
    # sns.heatmap(B, cmap = 'pink_r')
    plt.matshow(B)
    plt.title('prediction grotas')
    # plt.savefig(f'plots_load/prediction_{idx}_model{model_to_use_date}_{time_now}.png')
    plt.show()


if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    main()