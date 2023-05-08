import json
import os
from pathlib import Path
from l2g_approach.script_extract_csv_from_models import get_r_data_names, get_array_w_relevant_data


def main():
    file_path = os.path.realpath(__file__)
    folder_path = Path(file_path).parent
    export_file = 'runs_data_compare.csv'
    export_file_coalesce = 'runs_data_compare_coalesce.csv'

    files = os.listdir("runs")
    files = [os.path.join(folder_path, "runs", file) for file in files]
    files.sort(key=os.path.getmtime, reverse=True)

    all_grotas_runs = []
    for file in files:
        if "run_grotas_compare" in file and "B" not in file and "sigma_p" not in file and "theta" not in file:
            try:
                with open(file) as fp:
                    json_data = json.load(fp)
                    json_data['file'] = file
                    json_data['date'] = file.split("run_grotas_compare_")[1].split(".")[0].split('_')[0]
                    all_grotas_runs.append(json_data)
            except Exception as e:
                print(f"Error on {file}")

    l2g_loads = os.listdir(os.path.join("l2g_approach", "plots_load"))
    files = [os.path.join(folder_path, "l2g_approach", "plots_load", file) for file in l2g_loads]
    files.sort(key=os.path.getmtime, reverse=True)

    all_l2g_runs = []
    for file in files:
        if "json" in file:
            try:
                with open(file) as fp:
                    json_data = json.load(fp)
                    json_data['file'] = file
                    all_l2g_runs.append(json_data)
            except Exception as e:
                print(f"Error at l2g file {file}")



    all_l2g_models = {}
    l2g_loads = os.listdir(os.path.join("l2g_approach", "plots"))
    files = [os.path.join(folder_path, "l2g_approach", "plots", file) for file in l2g_loads]

    for file in files:
        if "run_results_data" in file:
            try:
                with open(file) as fp:
                    json_data = json.load(fp)
                    date = file.split("_")[-1].split(".")[0]
                    json_data['file'] = file
                    all_l2g_models[date] = json_data
            except Exception as e:
                print(f"Error at l2g model {file}")



    # Show results
    runs_organized = []

    for grotas_run in all_grotas_runs:
        comparative_l2g = [run for run in all_l2g_runs if run['observations'] == grotas_run['date']]
        if not comparative_l2g:
            continue
        runs_organized.append({
            "MSE": grotas_run['MSE'],
            "N": grotas_run['N'],
            "SNR": grotas_run.get('SNR'),
            "method": "grotas",
            "observations_date": grotas_run['date'],
            "random": grotas_run.get('random'),
        })
        for run in comparative_l2g:
            runs_organized.append({
                "MSE": run['MSE'],
                "N": grotas_run['N'],
                "SNR": grotas_run.get('SNR'),
                "method": "l2g",
                "observations_date": grotas_run['date'],
                "l2g_model": run['net'],
                "l2g_samples": all_l2g_models[run['net']]['parameters'].get('num_samples'),
                "l2g_SNR": all_l2g_models[run['net']]['parameters'].get('SNR'),
                "l2g_epochs": all_l2g_models[run['net']]['parameters'].get('n_epochs'),
                "l2g_gmse": all_l2g_models[run['net']]['results'].get('gmse_average'),
                "l2g_pred_gmse_mean": all_l2g_models[run['net']]['results'].get('pred_gmse_mean'),
                "l2g_train_mse": all_l2g_models[run['net']]['results'].get('epoch_train_mse',[None])[-1],
                "random": grotas_run.get('random'),
                "full_l2g_data": all_l2g_models[run['net']],
            })
            # for k, v in runs_organized[-1].items():
            #     print(f"    {k}: {v}")


    runs_by_observations = {}
    for run in runs_organized:
        if run['observations_date'] not in runs_by_observations:
            runs_by_observations[run['observations_date']] = []
        runs_by_observations[run['observations_date']].append(run)


    observations_titles = []
    models = {}
    models_headers = {}
    i = 0
    for date, runs in runs_by_observations.items():
        # lines.append(f"; date={date}, SNR={runs[0]['SNR']}, N={runs[0]['N']}")
        observations_titles.append(f"date={date}, SNR={runs[0]['SNR']}, N={runs[0]['N']}, r={runs[0].get('random')}")
        for run in runs:
            if run['method'] == "grotas":
                model_name = "grotas"
                method = f"grotas"
            else:
                model_name = run['l2g_model']
                model_info_data = get_array_w_relevant_data(run['full_l2g_data'])
                model_info_titles = get_r_data_names()
                model_info_full = [f"{name}={data}" for name, data in zip(model_info_titles, model_info_data)]
                model_info = ",".join(model_info_full)
                method = f"l2g, model={run['l2g_model']}, model_info=({model_info})"
                if float(run["MSE"]) > 20 and model_name not in models:  # Filter so I don't get a billion results
                    continue
            if model_name not in models:
                models[model_name] = [""] * len(runs_by_observations)
            models[model_name][i] = run['MSE']
            models_headers[model_name] = method
        i+=1
    with open(export_file, 'w') as fp:
        fp.write(";" + ";".join(observations_titles) + '\n')
        for model_name, values in models.items():
            ending = ";".join([str(v) for v in values])
            fp.write(f"{models_headers[model_name]};{ending}\n")

    runs_coalesced = {}
    observations_titles = []
    models = {}
    models_headers = {}
    i = 0
    runs_sorted_by_date = [(date, runs) for date, runs in runs_by_observations.items()]
    runs_sorted_by_date.sort(key=lambda x: x[0], reverse=True)
    for date, runs in runs_sorted_by_date:
        run_name = f"SNR={runs[0]['SNR']}, N={runs[0]['N']}, r={runs[0].get('random')}"
        if run_name not in runs_coalesced:
            runs_coalesced[run_name] = []
        runs_coalesced[run_name].extend(runs)
    for name, runs in runs_coalesced.items():
        observations_titles.append(name)
        for run in runs:
            if run['method'] == "grotas":
                model_name = "grotas"
                method = f"grotas"
            else:
                model_name = run['l2g_model']
                model_info_data = get_array_w_relevant_data(run['full_l2g_data'])
                model_info_titles = get_r_data_names()
                model_info_full = [f"{name}={data}" for name, data in zip(model_info_titles, model_info_data)]
                model_info = ",".join(model_info_full)
                method = f"l2g, model={run['l2g_model']}, model_info=({model_info})"
                if float(run["MSE"]) > 20:  # Filter so I don't get a billion results
                    continue
            if model_name not in models:
                models[model_name] = [""] * len(runs_coalesced)
            if models[model_name][i] == "":
                models[model_name][i] = run['MSE']
            models_headers[model_name] = method
        i+=1
    with open(export_file_coalesce, 'w') as fp:
        fp.write(";" + ";".join(observations_titles) + '\n')
        for model_name, values in models.items():
            ending = ";".join([str(v) for v in values])
            fp.write(f"{models_headers[model_name]};{ending}\n")


if __name__ == "__main__":
    main()

