import csv
import matplotlib.pyplot as plt
from datetime import datetime
from script_run_grotas_l2g_test import models as l2g_models


name = []
model_date = []
pred_gmse_mean = []
num_samples = []
n_hid = []
num_unroll = []

model_date_marked = []
pred_gmse_mean_marked = []
num_samples_marked = []
n_hid_marked = []
num_unroll_marked = []

with open("l2g_approach\\l2g_models_train.csv") as fp:
    csv_file = csv.reader(fp, delimiter=';', quotechar='"', )
    for i, row in enumerate(csv_file):
        if i != 0:
            if float(row[2]) > 5:
                continue
            name.append(row[0])
            date = row[0].split("_")[-1].split(".")[0]

            unix_time = datetime.strptime(date, "%Y%m%d%H%M%S")
            unix_time = unix_time.timestamp()
            model_date.append(unix_time)
            pred_gmse_mean.append(float(row[2]))
            num_samples.append(int(row[5]))
            n_hid.append(int(row[8]))
            num_unroll.append(int(row[7]))

            if date in l2g_models:
                model_date_marked.append(unix_time)
                pred_gmse_mean_marked.append(float(row[2]))
                num_samples_marked.append(int(row[5]))
                n_hid_marked.append(int(row[8]))
                num_unroll_marked.append(int(row[7]))




fig = plt.figure()
ax = plt.gca()
ax.scatter(model_date,pred_gmse_mean, c='blue', marker='x')
ax.scatter(model_date_marked,pred_gmse_mean_marked,  s=80, facecolors='none', edgecolors='r')
plt.grid(True)
# ax.scatter(data['o_value'] ,data['time_diff_day'] , c='blue', alpha=0.05, edgecolors='none')
ax.set_yscale('log')
# ax.set_xscale('log')
# plt.legend(['Puntos proyectados clase 1', 'Puntos proyectados clase 2', 'Puntos clase 1', 'Puntos clase 2'])
# plt.savefig('plotClase2020727.png')
plt.show()

def get_average(num_samples, sample_values):
    averages_lists = {}
    for i, value in enumerate(num_samples):
        if value not in averages_lists:
            averages_lists[value] = []
        averages_lists[value].append(i)

    averages = {}
    for value, all in averages_lists.items():
        all_values = [sample_values[x] for x in all]
        averages[value] = sum(all_values)/len(all_values)
    
    end_values = []
    end_avgs = []
    for value, avg in averages.items():
        end_values.append(value)
        end_avgs.append(avg)

    return end_values, end_avgs


num_samples_avg, avgs = get_average(num_samples,pred_gmse_mean)

fig = plt.figure()
ax = plt.gca()
ax.scatter(num_samples,pred_gmse_mean, c='blue', marker='x')
ax.scatter(num_samples_marked,pred_gmse_mean_marked,  s=80, facecolors='none', edgecolors='r')
ax.scatter(num_samples_avg,avgs,   s=80, edgecolors='green', marker='*')
plt.grid(True)
# ax.scatter(data['o_value'] ,data['time_diff_day'] , c='blue', alpha=0.05, edgecolors='none')
ax.set_yscale('log')
# ax.set_xscale('log')
# plt.legend(['Puntos proyectados clase 1', 'Puntos proyectados clase 2', 'Puntos clase 1', 'Puntos clase 2'])
# plt.savefig('plotClase2020727.png')
plt.show()

num_samples_avg, avgs = get_average(n_hid,pred_gmse_mean)
fig = plt.figure()
ax = plt.gca()
ax.scatter(n_hid,pred_gmse_mean, c='blue', marker='x')
ax.scatter(n_hid_marked,pred_gmse_mean_marked,  s=80, facecolors='none', edgecolors='r')
ax.scatter(num_samples_avg,avgs,   s=80, edgecolors='green', marker='*')
plt.grid(True)
# ax.scatter(data['o_value'] ,data['time_diff_day'] , c='blue', alpha=0.05, edgecolors='none')
ax.set_yscale('log')
# ax.set_xscale('log')
# plt.legend(['Puntos proyectados clase 1', 'Puntos proyectados clase 2', 'Puntos clase 1', 'Puntos clase 2'])
# plt.savefig('plotClase2020727.png')
plt.show()

num_samples_avg, avgs = get_average(num_unroll,pred_gmse_mean)
fig = plt.figure()
ax = plt.gca()
ax.scatter(num_unroll,pred_gmse_mean, c='blue', marker='x')
ax.scatter(num_unroll_marked,pred_gmse_mean_marked,  s=80, facecolors='none', edgecolors='r')
ax.scatter(num_samples_avg,avgs,   s=80, edgecolors='green', marker='*')
plt.grid(True)
# ax.scatter(data['o_value'] ,data['time_diff_day'] , c='blue', alpha=0.05, edgecolors='none')
ax.set_yscale('log')
# ax.set_xscale('log')
# plt.legend(['Puntos proyectados clase 1', 'Puntos proyectados clase 2', 'Puntos clase 1', 'Puntos clase 2'])
# plt.savefig('plotClase2020727.png')
plt.show()

