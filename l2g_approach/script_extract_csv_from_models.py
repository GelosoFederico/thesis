import json
import os

files = os.listdir("plots")
export_file = "l2g_models_train.csv"

print_things = False
csv = True


def important_data_from_run(run) -> str:
    param = run['parameters']
    return f"N={param.get('num_samples')}, un={param.get('num_unroll')}, SNR={param.get('SNR')}, eps={param.get('n_epochs')}, knp = {param['graph_hyper']['k']},{param['graph_hyper']['n_subnets']},{param['graph_hyper']['p_rewire']}"


def get_r_data_names() -> list:
    return [
        "file",
        "gmse_average",
        "pred_gmse_mean",
        "f_score_average",
        "epoch_train_mse",
        "num_samples",
        "num_signals",
        "num_unroll",
        "n_hid",
        "n_latent",
        "n_nodeFeat",
        "n_graphFeat",
        "lr",
        "lr_decay",
        "n_epochs",
        "SNR",
        "graph_hyperk",
        "graph_hypern_subnets",
        "graph_hyperp_rewire",
    ]


def get_array_w_relevant_data(run) -> list:
    return [
        run['file'],
        run["results"]['gmse_average'],
        run["results"]['pred_gmse_mean'],
        run["results"]['f_score_average'],
        run["results"].get('epoch_train_mse', [None])[-1],
        run["parameters"]['num_samples'],
        run["parameters"]['num_signals'],
        run["parameters"]['num_unroll'],
        run["parameters"]['n_hid'],
        run["parameters"]['n_latent'],
        run["parameters"]['n_nodeFeat'],
        run["parameters"]['n_graphFeat'],
        run["parameters"]['lr'],
        run["parameters"]['lr_decay'],
        run["parameters"]['n_epochs'],
        run["parameters"].get('SNR'),
        run["parameters"].get('graph_hyper', {}).get('k'),
        run["parameters"].get('graph_hyper', {}).get('n_subnets'),
        run["parameters"].get('graph_hyper', {}).get('p_rewire'),
    ]


def get_csv_like(arr):
    center = '";"'.join([str(x) for x in arr])
    return f'"{center}"'


def main():
    all_datas = []
    num_files = 10

    for file in files:
        if "run_results_data_rt_" in file or "run_results_data_random_rt_" in file:
            with open(os.path.join("plots", file)) as fp:
                json_data = json.load(fp)
                json_data['file'] = file
                all_datas.append(json_data)

    if print_things:
        print("By gmse_average")
        all_datas.sort(key=lambda x: float(x["results"]['gmse_average']))

        n = 0
        for dat in all_datas:
            print(dat['file'])
            print(dat["results"]['gmse_average'])
            print(important_data_from_run(dat))
            if "202304" in dat['file']:
                n+=1
                if n > num_files:
                    break


        print("By pred_gmse_mean")
        all_datas.sort(key=lambda x: float(x["results"]['pred_gmse_mean']))

        n = 0
        for dat in all_datas:
            print(dat['file'])
            print(dat["results"]['pred_gmse_mean'])
            print(important_data_from_run(dat))
            if "202304" in dat['file']:
                n+=1
                if n > num_files:
                    break

        print("By fscore")

        all_datas.sort(key=lambda x: -float(x["results"]['f_score_average']))

        n = 0
        for dat in all_datas:
            print(dat['file'])
            print(dat["results"]['f_score_average'])
            print(important_data_from_run(dat))
            if "202304" in dat['file']:
                n+=1
                if n > num_files:
                    break

    if csv:
        all_datas.sort(key=lambda x: float(x["results"]['pred_gmse_mean']))

        n = 0
        lines = []
        lines.append(get_csv_like(get_r_data_names()))
        for dat in all_datas:
            try:
                lines.append(get_csv_like(get_array_w_relevant_data(dat)))
            except Exception as e:
                print(e)
        with open(export_file, "w") as fp:
            for line in lines:
                fp.write(f"{line}\n")


if __name__ == "__main__":
    main()
