import time
from os import stat
import os
import subprocess
from pathlib import Path
from l2g_approach.l2g_script_utils import get_all_new_models

# l2g
models = [
    '20230318204755',
    '20230321004141',
    '20230322195931',
    '20230324044021',
    '20230324212012',
    '20230325040233',
    '20230325072623',
    '20230325231108',
    '20230326025017',
    '20230326110113',
    '20230326182042',
    '20230329084633',
    '20230401173804',
    '20230403033132',
    '20230403094638',
    '20230403135332',
    '20230403180128',
    '20230404075955',
    '20230404093313',
    '20230404113603',
    '20230404152507',
    '20230405085104',
    '20230408062300',
    '20230408234228',
    '20230410220452',
    '20230414032627',
    '20230415230630',
    '20230416003912',
    '20230416022604',
    '20230416033828',
    '20230416045240',
    '20230416164414',
    '20230416175429',
    '20230417024523',
    '20230417185252',
    '20230417191033',
    '20230420221106',
    '20230421064728',
    '20230421213953',
    '20230423234547',
    '20230424085150',
    '20230425001357',
    '20230426024535',
    '20230426131649',
    '20230427012146',
    '20230428030422',
    '20230428210711',
    '20230429155642',
    '20230430032000',
]

# models_restricted = [
#     "20230404113603",
#     "20230405085104",
#     "20230414032627",
#     "20230416022604",
#     "20230416033828",
#     "20230416045240",
#     "20230416164414",
#     "20230416175429",
# ]

# models = models_restricted
# models = get_all_new_models()

## grotas
n_samples = 3000
SNR = 40
# SNRs = [5,10,20,40,60,80]
# all_samples = [30, 300, 500, 1000, 2000, 3000]
SNRs = [20, 40, 60, 80]
all_samples = [1000, 2000, 3000]
# SNRs = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
# all_samples = [2000]
randoms = [True, False]

def main(models):
    n_runs = len(SNRs) * len(all_samples) * len(randoms) * len(models)
    i = 0
    durations_run = []
    for SNR in SNRs:
        for n_samples in all_samples:
            for random in randoms:

                t0_whole_group = time.time()

                conda_command = "conda activate cvxpy_env"

                parameters = {
                    'SNR': SNR,
                    'N': n_samples,
                }
                if random:
                    parameters['random'] = None
                grotas_command = "python GrotasCompare.py "
                grotas_command += " ".join([f"--{k} {v}" if v else f"--{k}" for k, v in parameters.items()])

                command = f"{conda_command} && {grotas_command}"

                print(f"Running {command}")
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                process.wait()
                print(process.returncode)
                print(process.stdout.read().decode())


                file_path = os.path.realpath(__file__)
                folder_path = Path(file_path).parent

                files = os.listdir("runs")
                files = [os.path.join(folder_path, "runs", file) for file in files]
                files.sort(key=os.path.getmtime, reverse=True)

                for file in files:
                    if "run_grotas_compare_" in file:
                        date = file.split("run_grotas_compare_")[1].split(".")[0].split('_')[0]
                        break

                print(date)

                models = list(set(models))
                models.sort()
                for model in models:
                    t0_run = time.time()
                    parameters = {
                        "observations_to_use_date": date,
                        "model_to_use_date": model
                    }

                    cd_command = "cd l2g_approach"

                    l2g_load_command = "venv\\scripts\\python l2g_load.py "
                    l2g_load_command += " ".join([f"--{k} {v}" for k, v in parameters.items()])

                    command = f"{cd_command} && {l2g_load_command}"

                    print(f"Running {command}")
                    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
                    process.wait()
                    print(process.returncode)
                    print(process.stdout.read().decode())
                    duration_run = time.time() - t0_run
                    print(f"Run took {duration_run:.4f} s")
                    durations_run.append(duration_run)
                    avg_run = sum(durations_run) / (i+1)
                    print(f"avg run: {avg_run:.4f}")
                    print(f"Run {i}/{n_runs}")
                    i += 1

                duration_group = time.time() - t0_whole_group
                print(f"Group took {duration_group:.4f} s")


if __name__ == "__main__":
    main(models)
