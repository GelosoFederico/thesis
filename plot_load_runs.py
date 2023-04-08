import csv
import matplotlib.pyplot as plt

def transform_comma_separated_to_dict(string):
    this_thing = {}
    values = string.split(",")
    for value in values:
        try:
            name, val = value.split("=")
            this_thing[name.strip()] = val.strip()
        except Exception:
            pass
    return this_thing


def merge_according_to(values, order):
    new_values = []
    for limits in order:
        possible_values = [x for x in values[limits[0]:limits[1]] if x]
        if possible_values:
            new_values.append(min(possible_values))
        else:
            new_values.append(None)
    return new_values
    

def main():
    models = {}
    runs = {}
    with open("test_runs_data.csv") as file:
        csv_file = csv.reader(file, delimiter=';', quotechar='"')
        for i, row in enumerate(csv_file):
            if i == 0:
                for j, elem in enumerate(row):
                    if not elem:
                        continue
                    this_run = transform_comma_separated_to_dict(elem)
                    runs[j] = {"run": this_run}
                    print(runs[j])
            
            else:
                for j, elem in enumerate(row):
                    if j == 0:
                        if '(' in elem:
                            model_name, model_data = elem.split('(')
                            model_name = transform_comma_separated_to_dict(model_name)['model']
                            model_data = model_data.split(')')[0]
                            model_data = transform_comma_separated_to_dict(model_data)
                            model_data['name'] = model_name
                        else:
                            model_name = elem
                            model_data = {"name": elem}
                        # print(model_data)
                        models[model_name] = model_data
                    else:
                        runs[j][model_name] = elem

    runs_random_values = {run['run']['r'] for run in runs.values()}
    runs_N_values = {run['run']['N'] for run in runs.values()}
    print(runs_N_values)
    print(runs_random_values)
    run_groups = {}
    for N in runs_N_values:
        for random in runs_random_values:
            run_groups[f"{N}-{random}"] = [run for run in runs.values() if run['run']['N'] == N and run['run']['r'] == random]
    

    for conj in run_groups.values():
        conj = [x for x in conj if x['run']['SNR'] != "None"]
        conj.sort(key=lambda v: float(v['run']['SNR']))
        # print(conj)

        SNR_values = [float(x['run']['SNR']) for x in conj if x['run']['SNR']]
        runners_values = {}
        for run in conj:
            for name, values in run.items():
                if name == "run":
                    continue
                if name not in runners_values:
                    runners_values[name] = []
                runners_values[name].append(float(values) if values else None)
        print(SNR_values)
        for name, values in runners_values.items():
            print(f"{name}: {values}")

        if len(SNR_values) < 2:
            continue

        # Let only be one of each SNR
        values_to_merge = []
        last_value = None
        for i, value in enumerate(SNR_values):
            if value != last_value:
                if values_to_merge:
                    values_to_merge[-1][1] = i
                values_to_merge.append([i, None])
            last_value = value
        
        values_to_merge[-1][1] = len(SNR_values)
        SNR_values = merge_according_to(SNR_values, values_to_merge)
        runners_values_merged = {k: merge_according_to(v, values_to_merge) for k,v in runners_values.items()}

        fig = plt.figure()
        for val in runners_values_merged.values():
            plt.plot(SNR_values, val)
        plt.ylabel('MSE')
        plt.xlabel('')
        plt.ylim([0,5])
        # plt.axis(ylim=(0, 5))
        # fig.update_layout(yaxis2 = dict(range=[0, 5]))

        plt.show()
        # plt.savefig('plots/GMSE_{}_{}.png'.format(graph_type, time_now))

        # print(runners_values)
    # for _, conj in run_groups.items():
    #     print(conj)

        


if __name__ == "__main__":
    main()