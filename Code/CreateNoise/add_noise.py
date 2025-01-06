import h5py
import numpy as np
import random
import csv
from sklearn import metrics
import os

bia = 7
Miu = 0.3
Sigma = 0.1
model_ver = 1
# we set the task for 32 regions lasting for 120 time slots
sample = [bia + 32 * i for i in range(32)]
tttime = 120
# the path of prediction result, we use m_ver1 as an example
path = '../Predict/scripts/model/bj_taxi_result/m_ver'
path += str(model_ver) + '/'

f = h5py.File('../Predict/data/BJ16_In.h5', 'r')
read = f['data'][()]
taxi_in = read[:, :, 0]
zip = taxi_in[-tttime:, sample]

f2 = h5py.File(path + 'bj_taxi.h5', 'r')
pred = f2['pred'][()]
pred = pred[-tttime:, sample]
# r2 = metrics.r2_score(zip.ravel(), pred.ravel())
# print('r2=', r2)
# np.random.seed(0)
# random.seed(0)
# we set 100 users and everyone submit sensing data for 30 times
dpp = np.array([30] * 100)
# choose 10 malicious MUs in 100 total users
b_worker = random.sample(range(100), 10)
u_d = [['user', 'time slot', 'region', 'ground truth', 'sensing traffic', 'prediction', 'noise(pp)']]
u_n = [['user', 'miu', 'sigma', 'noise(pp)']]

start = 2
end = 6  # create how much simulation datasets
for exp in range(start, end + 1):

    out = '../Result/origin/miu=' + str(Miu) + '_sigma=' + \
          str(Sigma) + '/simulate_data' + '/experiment' + str(exp) + '/'
    if not os.path.exists(out):
        os.makedirs(out)

    for i in range(100):
        # sample time slot number and area number for every user
        time = random.sample(range(0, tttime), dpp[i])
        area = np.random.randint(0, 32, dpp[i])
        mu_i = 0
        sig_i = 0
        # if user i is a malicious MU, add noise to his data
        if i in b_worker:
            mu_i = Miu
            sig_i = Sigma
        totalnoise = 0

        for j in range(dpp[i]):
            noise = random.gauss(mu_i, sig_i)
            if zip[time[j], area[j]] == 0:
                if abs(noise) >= 0.2:
                    u_d.append([i + 1, time[j], area[j], zip[time[j], area[j]], round(zip[time[j], area[j]]) + 1,
                                round(pred[time[j], area[j]]), noise])
                    continue
            u_d.append(
                [i + 1, time[j], area[j], zip[time[j], area[j]], round(zip[time[j], area[j]] * (1 + noise)),
                 round(pred[time[j], area[j]]), noise])
            totalnoise += noise
        u_n.append([i + 1, mu_i, sig_i, totalnoise / dpp[i]])

    with open(out + 'data.csv', 'w', encoding='utf-8-sig', newline='') as t:
        writer = csv.writer(t)
        writer.writerows(u_d)
        t.close()

    with open(out + 'noise.csv', 'w', encoding='utf-8-sig', newline='') as t2:
        writer = csv.writer(t2)
        writer.writerows(u_n)
        t2.close()
