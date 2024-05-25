import csv
import pandas as pd
from sklearn import metrics
import numpy as np

alpha = 0.018
beta = 0.06
gamma = 0.5


def updataR(R, q):
    if q >= gamma:
        k1 = alpha
        k2 = 1 / (1 - gamma)
    else:
        k1 = beta
        k2 = 1 / gamma
    return 0.5 + (k1 * (q - gamma) * k2 + (1 - k1) * (2 * R - 1)) * 0.5


def pred_based(path, rep):
    list = {}
    truth = []
    pred = []
    totalT = 0
    with open(path + 'data.csv', encoding='utf-8-sig') as f:
        for i, data in enumerate(csv.reader(f)):
            if i == 0:
                continue
            truth.append(float(data[3]))
            pred.append(float(data[5]))
            list.setdefault(int(data[0]), [])
            totalT = max(totalT, int(data[1]))
            list[int(data[0])].append([int(data[1]), int(data[2]), float(data[3]), float(data[4]), float(data[5])])
        f.close()
    b_worker = []
    with open(path + 'noise.csv', encoding='utf-8-sig') as f2:
        for i, data in enumerate(csv.reader(f2)):
            if i == 0:
                continue
            if float(data[3]) != 0:
                b_worker.append(int(data[0]))
        f2.close()
    te = 0
    n_te = 0
    for time in range(totalT + 1):
        errorlist = {}
        telist = {}
        for user in list.keys():
            for data in list[user]:
                # data will be added to the list only when it is in this time slot
                if time == data[0]:
                    error = (data[3] - data[4]) / data[4] if data[4] != 0 else data[3] - data[4]
                    errorlist[user] = error
                    telist[user] = abs(data[3] - data[2])
        max_error = abs(max(errorlist.items(), key=lambda x: abs(x[1]))[1])
        min_error = 0
        delta = max_error - min_error if max_error != 0 else 1
        for user in errorlist.keys():
            q = 1 - (abs(errorlist[user]) - min_error) / delta
            rep[user] = updataR(rep[user], q)
            if q > 0.5:
                te += telist[user]
                n_te += 1

    slist = sorted(rep.items(), key=lambda x: x[1])
    wrong = 0
    for i in range(len(b_worker)):
        if slist[i][0] not in b_worker:
            wrong += 1
    precision = (len(rep) - len(b_worker) - wrong) / (len(rep) - len(b_worker))
    recall = (len(rep) - len(b_worker) - wrong) / (len(rep) - len(b_worker))
    F1 = 2 * precision * recall / (precision + recall)
    bad = 0
    good = 0
    for key in rep.keys():
        if key in b_worker:
            bad += rep[key]
        else:
            good += rep[key]
    distance = good / (len(list) - len(b_worker)) - bad / len(b_worker)
    acc = (len(b_worker) - wrong) / len(b_worker)
    mae = metrics.mean_absolute_error(truth, pred)
    return F1, distance, mae, te / n_te


if __name__ == '__main__':
    type = 'origin'
    # type = 'accident'
    # type = 'sparsity'
    miu = 0.3
    sigma = 0.1
    path = '../Result/' + type + '/miu=' + str(miu) + '_sigma=' + str(sigma)
    exp_start = 1
    exp_stop = 1
    col1 = []
    col2 = []
    col3 = []
    init_rep = {}
    for key in range(1, 101):
        init_rep[key] = 0.5
    for ver in range(exp_start, exp_stop + 1):
        rpath = path + '/simulate_data/experiment' + str(ver) + '/'
        F1, distance, mae, te = pred_based(rpath, init_rep)
        col1.append(F1)
        col2.append(distance)
        col3.append(te)
    print(np.mean(col1))
    print(np.mean(col2))
    print(np.mean(col3))
    outpath = path + '/result.csv'
    result = pd.read_csv(outpath)
    result['CNB F1'] = col1
    result['CNB rep dis'] = col2
    result['CNB noise'] = col3
    result.to_csv(outpath, index=False, encoding='utf-8-sig')
