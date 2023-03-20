import argparse
from datetime import datetime
import json
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.utils import gmse_loss_batch, gmse_loss_batch_mean, layerwise_gmse_loss
from src.utils_data import data_loading, rotate_matrix, get_z_and_w_gt, get_a_from_matrix, get_z_from_samples
from src.utils_grotas import IEEE57_b_matrix, get_b_matrix_from_positive_zero_diagonal, F_score
from src.models import load_l2g_from_disk


def MSE_matrix(matrix_real, matrix_est):
    diff_matrix = matrix_real - matrix_est
    mse = np.trace(diff_matrix @ diff_matrix.T)
    if mse != np.real(mse):
        print(mse)
        mse = np.real(mse)
    return mse


def main(model_to_use_date, observations_to_use_date):
    time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = load_l2g_from_disk(f"saved_model\\net_{model_to_use_date}").to(device)

    batch_size = 32
    # data_dir = f"data\\dataset_rt_nested_57nodes_rt_nested_{model_to_use_date}.pickle"
    # train_loader, val_loader, test_loader = data_loading(data_dir, batch_size=batch_size)

    # Use the IEEE 57 options
    num_samples = batch_size
    matrixes = []
    for i in range(num_samples):
        new_ieee57, a = IEEE57_b_matrix()
        if i != 0:
            new_ieee57 = rotate_matrix(new_ieee57)
        a = get_a_from_matrix(new_ieee57)
        # plt.figure()
        # sns.heatmap(a, cmap = 'pink_r')
        # plt.title('prediction')
        # plt.show()
        # print("new_ieee57.shape")
        # print(new_ieee57.shape)
        # print(a.shape)
        matrixes.append((new_ieee57, a))

    # final_w = []
    all_z = []

    # for mat, a in matrixes:
        # plt.figure()
        # sns.heatmap(a, cmap = 'pink_r')
        # plt.title('prediction')
        # plt.show()
        # np.fill_diagonal(mat, 0)
        # TODO this is the one line to use if not reloading
        # z, W_GT, samples, states = get_z_and_w_gt(mat, mat.shape[0], 300, a)

        # all_z.append(z)
        # final_w.append(W_GT)
    with open(f"..\\data\\observations_{observations_to_use_date}.pickle", 'rb') as handle:
        dataset = pickle.load(handle)
    for mat, a in matrixes:
        all_z.append(get_z_from_samples(np.matrix(dataset['samples']).T))

    w = []
    for mat in matrixes:
        # other_mat = mat[0].copy()
        # np.fill_diagonal(other_mat, 0)
        w.append(mat[0])
    data = TensorDataset(torch.Tensor(all_z), torch.Tensor(w))
    test_loader = DataLoader(data, batch_size=batch_size)

    num_unroll = net.layers

    for z, w_gt_batch in test_loader:
        test_loss = []

        z = z.to(device)

        w_list = net.validation(z, threshold=1e-2)
        w_pred = torch.clamp(w_list[:, num_unroll - 1, :], min=0)

        # loss_mean = gmse_loss_batch_mean(w_pred, w_gt_batch)
        # loss_pred = gmse_loss_batch(w_pred, w_gt_batch)

        # layer_loss_batch = torch.stack([layerwise_gmse_loss(w_list[i, :, :], w_gt_batch[i, :]) for i in range(batch_size)])

    # idx = 10
    # plt.figure()
    # sns.heatmap(squareform(w_pred[idx,:].detach().cpu().numpy()), cmap = 'pink_r')
    # plt.title('prediction')
    # plt.savefig(f'plots_load/prediction_{idx}_{time_now}.png')
    # plt.show()

    # plt.figure()
    # mat_to_plot = w_gt_batch[idx,:].detach().cpu().numpy()
    # mat_to_plot = -mat_to_plot
    # np.fill_diagonal(mat_to_plot, 0)
    # sns.heatmap(mat_to_plot, cmap = 'pink_r')
    # plt.title('groundtruth')
    # plt.savefig(f'plots_load/groundtruth_{idx}_{time_now}.png')
    # plt.show()

    # plt.figure()
    # sns.heatmap(matrixes[idx][1], cmap = 'pink_r')
    # plt.title('groundtruth2')
    # plt.savefig(f'plots_load/groundtruth2_{idx}_{time_now}.png')
    # plt.show()


    idx = 0
    pred = squareform(w_pred[idx,:].detach().cpu().numpy())
    pred[np.abs(pred) < 0.1] = 0
    B = get_b_matrix_from_positive_zero_diagonal(pred)
    # B = get_b_matrix_from_positive_zero_diagonal(w_gt_batch[1, :].detach().cpu().numpy())

    # L = np.linalg.inv(B.T@B)@B.T
    state_covariance = np.eye(B.shape[0])
    L_grotas = state_covariance @ B @ np.linalg.pinv(B.T@state_covariance@B+np.eye(B.shape[0]))
    error_from_state_to_sample = []
    # error_from_sample_to_state = []
    error_from_sample_to_state_grotas = []
    error_from_state_to_sample_real = []
    sum_states = []
    sum_samples = []
    # for i in range(samples.shape[1]):
    #     sum_states.append(np.sum(states[:, i]**2))
    #     sum_samples.append(np.sum(samples[:, i]**2))
    #     error_from_state_to_sample_real.append(np.sum(((B @ states[:, i]) - samples[:, i])**2))
    #     error_from_state_to_sample.append(np.sum(((B @ states[:, i]) - samples[:, i])**2))
    #     # error_from_sample_to_state.append(np.sum(((L @ samples[:, i]) - states[:, i])**2))
    #     error_from_sample_to_state_grotas.append(np.sum(((L_grotas @ samples[:, i]) - states[:, i])**2))
    # print(sum_states)
    # print(sum_samples)
    # print(error_from_state_to_sample)
    # # print(error_from_sample_to_state)
    # print(error_from_sample_to_state_grotas)
    plt.figure()
    pred_mat = squareform(w_pred[idx,:].detach().cpu().numpy())
    # sns.heatmap(B, cmap = 'pink_r')
    plt.matshow(B)
    plt.title('prediction')
    plt.savefig(f'plots_load/prediction_{idx}_model{model_to_use_date}_{time_now}.png')
    # plt.show()

    plt.figure()
    mat_to_plot = w_gt_batch[idx,:].detach().cpu().numpy()
    mat_to_plot = -mat_to_plot
    np.fill_diagonal(mat_to_plot, 0)
    sns.heatmap(mat_to_plot, cmap = 'pink_r')
    plt.title('groundtruth')
    plt.savefig(f'plots_load/groundtruth_{idx}_model{model_to_use_date}_{time_now}.png')
    # plt.show()

    plt.figure()
    sns.heatmap(matrixes[idx][1], cmap = 'pink_r')
    plt.title('groundtruth2')
    plt.savefig(f'plots_load/groundtruth2_{idx}_model{model_to_use_date}_{time_now}.png')
    # plt.show()

    
    plt.figure()
    sns.heatmap(np.abs(mat_to_plot - pred_mat), cmap = 'pink_r')
    plt.title('difference')
    plt.savefig(f'plots_load/difference_{idx}_model{model_to_use_date}_{time_now}.png')
    # plt.show()

    plt.figure()
    sns.heatmap(np.abs(matrixes[idx][1] - get_a_from_matrix(pred_mat)), cmap = 'pink_r')
    plt.title('difference A')
    plt.savefig(f'plots_load/differenceA_{idx}_model{model_to_use_date}_{time_now}.png')
    # plt.show()

    # TODO we are doing only one matrix, we have 64
    MSE = MSE_matrix(mat_to_plot, pred_mat)
    fs = F_score(mat_to_plot, pred_mat)
    # MSE_states_total = MSE_states(observations, B, sigma_theta, sigma_est**2, states)
    test_result = {
        "net": model_to_use_date,
        "observations": observations_to_use_date,
        "MSE": MSE.item(),
        "F_score": fs.item()
    }
    with open(f"plots_load/run_{model_to_use_date}_with_{observations_to_use_date}.json", 'w') as fp:
        json.dump(test_result, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_to_use_date', default="20230215014040", type=str)
    parser.add_argument('--observations_to_use_date', default="20230320011919", type=str)
    parsed_args = parser.parse_args()
    main(parsed_args.model_to_use_date, parsed_args.observations_to_use_date)