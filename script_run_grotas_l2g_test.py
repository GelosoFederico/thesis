
from os import stat
import os
import subprocess
from pathlib import Path



## grotas
n_samples = 3000
SNR = 40
# SNRs = [5,10,20,40,60,80]
# all_samples = [30, 300, 500, 1000, 2000, 3000]
SNRs = [20, 40, 60, 80]
all_samples = [1000, 2000, 3000]
randoms = [True, False]

for SNR in SNRs:
    for n_samples in all_samples:
        for random in randoms:

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

            # l2g
            models = [
                "20230318204755",
                "20230321004141",
                "20230322195931",
                "20230324044021",
                "20230324144007",
                "20230324212012",
                "20230325040233",
                "20230325072623",
                "20230325141848",
                "20230325231108",
                "20230326025017",
                "20230326060225",
                "20230326110113",
                "20230326182042",
                "20230326211948",
                "20230329084633",
                "20230401070536",
                "20230401173804",
                "20230403033132",
                "20230403094638",
                "20230403135332",
                "20230403180128",
                "20230404032530",
                "20230404075955",
                "20230404113603",
                "20230404152507",
                "20230405085104",
                "20230408062300",
            ]
            models = list(set(models))
            for model in models:
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



