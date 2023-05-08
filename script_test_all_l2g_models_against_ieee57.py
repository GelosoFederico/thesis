import os
import json
import subprocess

# date=20230406180008, SNR=60.0, N=3000, r=false
observations = "20230406180008"
files = os.listdir("l2g_approach\\plots")
models = []
for file in files:

    if "run_results" in file:
        words = file.split(".")[0].split("_")
        for word in words:
            if word.startswith("2023"):
                models.append(word)

# models = ["20230404113603", "20230416022604", "20230416045240", "20230416164414"]

load_path = os.path.join("l2g_approach", "plots_load")
files_load = os.listdir(load_path)

for model in models:
    parameters = {
        "observations_to_use_date": observations,
        "model_to_use_date": model
    }
    model_in_loaded = [file_loaded for file_loaded in files_load if model in file_loaded]
    if model_in_loaded:
        models_w_obs = [m for m in model_in_loaded if observations in m]
        if models_w_obs:
            print(f"Model found in {models_w_obs[0]}")
            load_thing = os.path.join(load_path, models_w_obs[0])
            with open(load_thing) as fp:
                run = json.load(fp)
                print(run['MSE'])
            continue


    cd_command = "cd l2g_approach"

    l2g_load_command = "venv\\scripts\\python l2g_load.py "
    l2g_load_command += " ".join([f"--{k} {v}" for k, v in parameters.items()])

    command = f"{cd_command} && {l2g_load_command}"

    print(f"Running {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)
    print(process.stdout.read().decode())
