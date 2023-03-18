import subprocess
import random

parameters = {
'graph_type': 'rt_nested',
'num_unroll': 20,
'num_samples': 8064,
'num_signals': 3000,
'k': 2,
'n_subnets': 5,
'p_rewire': 0.4,
'lr': 0.01,
'lr_decay': 0.9999,
'n_epochs': 300,
}

# randomize
if random.random() > 0.8:
    parameters['num_unroll'] = round(parameters['num_unroll'] + parameters['num_unroll'] * random.uniform(-0.2, 0.2))
if random.random() > 0.8:
    parameters['num_signals'] = round(parameters['num_signals'] + parameters['num_signals'] * random.uniform(-0.2, 0.2))
if random.random() > 0.95:
    parameters['k'] = round(parameters['k'] + random.randint(-1,1))
if random.random() > 0.8:
    parameters['n_subnets'] = round(parameters['n_subnets'] + random.randint(-1,1))
if random.random() > 0.8:
    parameters['p_rewire'] = parameters['p_rewire'] + parameters['p_rewire'] * random.uniform(-0.2, 0.2)
if random.random() > 0.2:
    parameters['n_epochs'] = round(parameters['n_epochs'] + parameters['n_epochs'] * random.uniform(-0.6, 0.2))

command = "venv\\scripts\\python l2g_main.py "
command += " ".join([f"--{k} {v}" for k,v in parameters.items()])
print(command)
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()
print(process.returncode)