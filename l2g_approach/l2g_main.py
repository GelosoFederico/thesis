"""
This is from https://github.com/xpuoxford/L2G-neurips2021
with some changes to go with our rt nested graphs
"""
import json
from datetime import datetime
from numpy import average
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pickle

from src.models import *
from src.utils import *
from src.utils_data import *
from src.utils_grotas import F_score
import time
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    run_comments = []

    time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    # graph_type = 'WS'
    graph_type = 'rt_nested'
    # graph_type = 'ieee57'
    graph_size = 57 # default 50
    num_unroll = 20

    logging.basicConfig(filename='logs/L2G_{}_m{}_x{}.log'.format(graph_type, graph_size, num_unroll),
                        filemode='w',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    # generate synthetic WS graphs
    edge_type = 'lognormal'

    if graph_type == 'WS':
        graph_hyper = {'k': 5, # default k=5
                    'p': 0.4}
        num_samples = 16064 # default 8064
        num_signals = 3000 # default 3000

        data = generate_WS_parallel(num_samples=num_samples,
                                    num_signals=num_signals, # 3000 default
                                    num_nodes=graph_size,
                                    graph_hyper=graph_hyper,
                                    weighted=edge_type,
                                    weight_scale=True)
    elif graph_type == 'rt_nested':
        # TODO get true hyperparams
        graph_hyper = {
            'k': 2,
            'n_subnets': 5,
            'p_rewire': 0.4
        }
        num_samples=8064

        num_signals=3000
        # N = int(graph_size / graph_hyper['k'])
        # print(N)
        data = generate_rt_nested_parallel(num_samples=num_samples,
                                    num_signals=num_signals,
                                    num_nodes=graph_size,
                                    graph_hyper=graph_hyper,
                                    weighted=edge_type,
                                    weight_scale=True)
    else:
        num_samples=8064
        num_signals=3000
        graph_hyper = {}
        data = generate_57_ieee_parallel(num_samples=num_samples,
                                    num_signals=num_signals,
                                    num_nodes=graph_size,
                                    graph_hyper={},
                                    weighted=edge_type,
                                    weight_scale=True)

    if data['W'][0].shape[0] != graph_size:
        logger.info(f"Graphs created are of size {data['W'][0].shape[0]}. We expected {graph_size}, but we will use that")
        graph_size = data['W'][0].shape[0]

    with open('data/dataset_{}_{}nodes_{}_{}.pickle'.format(graph_type, graph_size, graph_type, time_now), 'wb') as handle:
        del data['samples']
        del data['states']
        pickle.dump(data, handle, protocol=4)


    # load data

    batch_size = 32

    data_dir = 'data/dataset_{}_{}nodes_{}_{}.pickle'.format(graph_type, graph_size, graph_type, time_now)
    train_loader, val_loader, test_loader = data_loading(data_dir, batch_size=batch_size)


    # for _, W, _, _ in test_loader:
    for _, W in test_loader:
        eg = torch_sqaureform_to_matrix(W, device='cpu')

    num_unroll = 20
    # graph_size = 50
    n_hid = 32
    n_latent = 16
    n_nodeFeat = 1
    n_graphFeat = 16

    lr = 1e-02
    lr_decay = 0.9999




    net = learn2graph(num_unroll, graph_size, n_hid,
                      n_latent, n_nodeFeat, n_graphFeat).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, lr_decay)

    logging.info(net)

    # Training:
    n_epochs = 120 # 300 default

    run_values = {
        'graph_algorithm': graph_type,
        'graph_hyper': graph_hyper,
        'num_samples': num_samples,
        'num_signals': num_signals,
        'num_unroll': num_unroll,
        'n_hid': n_hid,
        'n_latent': n_latent,
        'n_nodeFeat': n_nodeFeat,
        'n_graphFeat': n_graphFeat,
        'lr': lr,
        'lr_decay': lr_decay,
        'n_epochs': n_epochs,
    }
    with open(f"plots/run_values_{graph_type}_{time_now}.txt",'w') as f:
        json.dump(run_values, f, indent=4)
        # f.writelines(run_lines)
    dur = []

    epoch_train_gmse = []
    epoch_val_gmse = []

    for epoch in range(n_epochs):

        train_unrolling_loss, train_vae_loss, train_kl_loss, train_gmse, val_gmse = [], [], [], [], []

        t0 = time.time()

        net.train()
        # for z, w_gt_batch, sampl, state in train_loader:
        for z, w_gt_batch in train_loader:
            z = z.to(device)
            w_gt_batch = w_gt_batch.to(device)
            this_batch_size = w_gt_batch.size()[0]

            optimizer.zero_grad()
            w_list, vae_loss, vae_kl, _ = net.forward(z, w_gt_batch, threshold=1e-04, kl_hyper=1)  # default threshold 1e-04

            unrolling_loss = torch.mean(
                torch.stack([acc_loss(w_list[i, :, :], w_gt_batch[i, :], dn=0.9) for i in range(batch_size)])
            )

            loss = unrolling_loss + vae_loss
            loss.backward()
            optimizer.step()

            w_pred = w_list[:, num_unroll - 1, :]
            gmse = gmse_loss_batch_mean(w_pred, w_gt_batch)

            train_gmse.append(gmse.item())
            train_unrolling_loss.append(unrolling_loss.item())
            train_vae_loss.append(vae_loss.item())
            train_kl_loss.append(vae_kl.item())

        scheduler.step()

        net.eval()
        # for z, w_gt_batch, samples, states in val_loader:
        for z, w_gt_batch in val_loader:
            z = z.to(device)
            w_gt_batch = w_gt_batch.to(device)

            w_list = net.validation(z, threshold=1e-04)
            w_pred = torch.clamp(w_list[:, num_unroll - 1, :], min=0)
            loss = gmse_loss_batch_mean(w_pred, w_gt_batch)
            val_gmse.append(loss.item())
            # if epoch > n_epochs-2:
            #     print('test')
                # Ver como pasar de w a L

        dur.append(time.time() - t0)

        logging.info("Epoch {:04d} | lr: {:04.8f} | Time(s): {:.4f}".format(epoch + 1, scheduler.get_lr()[0], np.mean(dur)))
        logging.info("== train Loss <unroll: {:04.4f} | vae : {:04.4f} | kl : {:04.4f}>".format(np.mean(train_unrolling_loss),
                                                                                      np.mean(train_vae_loss),
                                                                                      np.mean(train_kl_loss)))
        logging.info("== gmse <train: {:04.4f} | val: {:04.4f}> ".format(np.mean(train_gmse), np.mean(val_gmse)))

        epoch_train_gmse.append(np.mean(train_gmse))
        epoch_val_gmse.append(np.mean(val_gmse))

    # validation loss:

    plt.figure()
    plt.plot(epoch_val_gmse)
    plt.ylabel('GMSE')
    plt.xlabel('epoch')
    plt.savefig('plots/validation_loss_{}_{}.png'.format(graph_type, time_now))

    # plt.show()


    for z, w_gt_batch in test_loader:
        test_loss = []

        z = z.to(device)
        w_gt_batch = w_gt_batch.to(device)
        this_batch_size = w_gt_batch.size()[0]

        adj_batch = w_gt_batch.clone()
        adj_batch[adj_batch > 0] = 1

        w_list = net.validation(z, threshold=1e-04)
        w_pred = torch.clamp(w_list[:, num_unroll - 1, :], min=0)

        loss_mean = gmse_loss_batch_mean(w_pred, w_gt_batch)
        loss_pred = gmse_loss_batch(w_pred, w_gt_batch)

        layer_loss_batch = torch.stack([layerwise_gmse_loss(w_list[i, :, :], w_gt_batch[i, :]) for i in range(batch_size)])


    loss_all_data = loss_pred.detach().cpu().numpy()
    final_pred_loss, final_pred_loss_ci, _, _ = mean_confidence_interval(loss_all_data, 0.95)
    logging.info('GMSE: {} +- {}'.format(final_pred_loss, final_pred_loss_ci))

    aps_auc = binary_metrics_batch(adj_batch, w_pred, device)
    logging.info('aps: {} +- {}'.format(aps_auc['aps_mean'], aps_auc['aps_ci']))
    logging.info('auc: {} +- {}'.format(aps_auc['auc_mean'], aps_auc['auc_ci']))

    layer_loss_mean = [mean_confidence_interval(layer_loss_batch[:,i].detach().cpu().numpy(), confidence=0.95)[0] for i in range(num_unroll)]
    layer_loss_mean_ci = [mean_confidence_interval(layer_loss_batch[:,i].detach().cpu().numpy(), confidence=0.95)[1] for i in range(num_unroll)]
    logging.info('layerwise test loss :{}'.format(layer_loss_mean))

    plt.figure()
    plt.plot(np.arange(1,len(layer_loss_mean)+1,1), layer_loss_mean)
    plt.xticks(np.arange(1,len(layer_loss_mean)+1,1))
    plt.ylabel('GMSE')
    plt.xlabel('number of unrolls/iterations')
    plt.savefig('plots/GMSE_{}_{}.png'.format(graph_type, time_now))
    # plt.show()

    result = {
        'epoch_train_gmse': epoch_train_gmse,
        'epoch_val_gmse': epoch_val_gmse,
        'pred_gmse_mean': final_pred_loss,
        'pred_gmse_mean_ci': final_pred_loss_ci,
        'auc_mean': aps_auc['auc_mean'],
        'auc_ci': aps_auc['auc_ci'],
        'aps_mean': aps_auc['aps_mean'],
        'aps_ci': aps_auc['aps_ci'],
        'layerwise_gmse_mean': layer_loss_mean,
        'layerwise_gmse_mean_ci ': layer_loss_mean_ci
    }


    result_path = 'saved_results/L2G_{}{}_unroll{}_{}_{}.pt'.format(graph_type,
                                                              graph_size,
                                                              num_unroll,
                                                              graph_type,
                                                              time_now)

    with open(result_path, 'wb') as handle:
        pickle.dump(result, handle, protocol=4)

    logging.info('results saved at: {}'.format(result_path))

    idx = 10
    plt.figure()
    sns.heatmap(squareform(w_pred[idx,:].detach().cpu().numpy()), cmap = 'pink_r')
    plt.title('prediction')
    plt.savefig('plots/prediction_{}_{}.png'.format(graph_type, time_now))
    # plt.show()

    plt.figure()
    sns.heatmap(squareform(w_gt_batch[idx,:].detach().cpu().numpy()), cmap = 'pink_r')
    plt.title('groundtruth')
    plt.savefig('plots/groundtruth_{}_{}.png'.format(graph_type, time_now))
    # plt.show()
    idx = 20
    plt.figure()
    sns.heatmap(squareform(w_pred[idx,:].detach().cpu().numpy()), cmap = 'pink_r')
    plt.title('prediction')
    plt.savefig('plots/prediction_{}_{}.png'.format(graph_type, time_now))
    # plt.show()

    plt.figure()
    sns.heatmap(squareform(w_gt_batch[idx,:].detach().cpu().numpy()), cmap = 'pink_r')
    plt.title('groundtruth')
    plt.savefig('plots/groundtruth_{}_{}.png'.format(graph_type, time_now))
    # plt.show()

    f_score_average = get_f_score_average(w_gt_batch, w_pred)
    gmse_average = get_gmse_average(w_gt_batch, w_pred)
    result['gmse_average'] = gmse_average
    result['f_score_average'] = f_score_average

    # We do this to ensure objects are json serializable. Mostly for float32.
    for k, v in result.items():
        try:
            result[k] = v.item()
        except Exception:
            try:
                result[k] = [x.item() for x in v]
            except Exception:
                pass

    whole_data = {
        "parameters": run_values,
        "results": result,
        "comments": run_comments
    }

    with open(f"plots/run_results_data_{graph_type}_{time_now}.json", 'w') as f:
        json.dump(whole_data, f, indent=4)

    net.save_to_disk(f"saved_model/net_{time_now}")


def get_f_score_average(groundtruth, prediction):
    all_scores = []
    for i, _ in enumerate(groundtruth):
        all_scores.append(F_score(squareform(groundtruth[i,:].detach().cpu().numpy()),
                          squareform(prediction[i,:].detach().cpu().numpy())))
    return average(all_scores)


def get_gmse_average(groundtruth, prediction):
    all_scores = []
    for i, _ in enumerate(groundtruth):
        all_scores.append(gmse_loss_batch_mean(groundtruth[i,:],
                          prediction[i,:]).detach().cpu().numpy())
    return average(all_scores)


if __name__ == "__main__":
    multiprocess.freeze_support()
    main()