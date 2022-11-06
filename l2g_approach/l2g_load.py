from datetime import datetime
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.utils import gmse_loss_batch, gmse_loss_batch_mean, layerwise_gmse_loss
from src.utils_data import data_loading, rotate_matrix, get_z_and_w_gt, get_a_from_matrix
from src.utils_grotas import IEEE57_b_matrix
from src.models import load_l2g_from_disk


def main():
    time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = load_l2g_from_disk("saved_model\\net_20221105121332").to(device)

    batch_size = 32
    data_dir = "data\\dataset_rt_nested_57nodes_rt_nested_20221105121332.pickle"
    train_loader, val_loader, test_loader = data_loading(data_dir, batch_size=batch_size)

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

    final_w = []
    all_z = []

    for mat, a in matrixes:
        # plt.figure()
        # sns.heatmap(a, cmap = 'pink_r')
        # plt.title('prediction')
        # plt.show()        
        z, W_GT = get_z_and_w_gt(mat, mat.shape[0], 30000, a)
        all_z.append(z)
        final_w.append(W_GT)

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
        w_gt_batch = w_gt_batch.to(device)
        this_batch_size = w_gt_batch.size()[0]

        adj_batch = w_gt_batch.clone()
        adj_batch[adj_batch > 0] = 1

        w_list = net.validation(z, threshold=1e-15)
        w_pred = torch.clamp(w_list[:, num_unroll - 1, :], min=0)

        # loss_mean = gmse_loss_batch_mean(w_pred, w_gt_batch)
        # loss_pred = gmse_loss_batch(w_pred, w_gt_batch)

        # layer_loss_batch = torch.stack([layerwise_gmse_loss(w_list[i, :, :], w_gt_batch[i, :]) for i in range(batch_size)])

    idx = 10
    plt.figure()
    sns.heatmap(squareform(w_pred[idx,:].detach().cpu().numpy()), cmap = 'pink_r')
    plt.title('prediction')
    plt.savefig(f'plots_load/prediction_{idx}_{time_now}.png')
    plt.show()

    plt.figure()
    mat_to_plot = w_gt_batch[idx,:].detach().cpu().numpy()
    mat_to_plot = -mat_to_plot
    np.fill_diagonal(mat_to_plot, 0)
    sns.heatmap(mat_to_plot, cmap = 'pink_r')
    plt.title('groundtruth')
    plt.savefig(f'plots_load/groundtruth_{idx}_{time_now}.png')
    plt.show()

    plt.figure()
    sns.heatmap(matrixes[idx][1], cmap = 'pink_r')
    plt.title('groundtruth2')
    plt.savefig(f'plots_load/groundtruth2_{idx}_{time_now}.png')
    plt.show()


    idx = 0
    plt.figure()
    sns.heatmap(squareform(w_pred[idx,:].detach().cpu().numpy()), cmap = 'pink_r')
    plt.title('prediction')
    plt.savefig(f'plots_load/prediction_{idx}_{time_now}.png')
    plt.show()

    plt.figure()
    mat_to_plot = w_gt_batch[idx,:].detach().cpu().numpy()
    mat_to_plot = -mat_to_plot
    np.fill_diagonal(mat_to_plot, 0)
    sns.heatmap(mat_to_plot, cmap = 'pink_r')
    plt.title('groundtruth')
    plt.savefig(f'plots_load/groundtruth_{idx}_{time_now}.png')
    plt.show()

    plt.figure()
    sns.heatmap(matrixes[idx][1], cmap = 'pink_r')
    plt.title('groundtruth2')
    plt.savefig(f'plots_load/groundtruth2_{idx}_{time_now}.png')
    plt.show()

if __name__ == "__main__":
    main()