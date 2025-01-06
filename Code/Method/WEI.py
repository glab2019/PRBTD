import csv
import os
import pandas as pd
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


def WEI(path, rep):
    list = {}
    pred = []
    yt = {}
    totalT = 0
    total_e = []
    with open(path + 'data.csv', encoding='utf-8-sig') as f:
        for i, data in enumerate(csv.reader(f)):
            if i == 0:
                continue
            pred.append(float(data[5]))
            totalT = max(totalT, int(data[1]))
            list.setdefault(int(data[0]), [])
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
    de_total_e = []
    for time in range(totalT + 1):
        errorlist = {}
        tempdata = []
        tempy = {}
        tempcnt = {}
        telist = {}
        totalw = 0
        for user in list.keys():
            totalw += rep[user]
            for data in list[user]:
                # data will be added to the list only when it is in this time slot
                if time == data[0]:
                    area = data[1]
                    tempy.setdefault(area, 0)
                    tempdata.append([user, data])
                    telist[user] = abs(data[3] - data[2])
                    total_e.append(abs(data[3] - data[2]))

        for user in list.keys():
            for data in list[user]:
                if time == data[0]:
                    area = data[1]
                    tempy[area] += data[3] * rep[user] / totalw
                    tempcnt.setdefault(area, 0)
                    tempcnt[area] += 1
                    tempdata.append([user, data])
                    telist[user] = abs(data[3] - data[2])

        # update the estimation based on average with weight
        for tarea in tempy.keys():
            yt.setdefault(tarea, tempy[tarea])
            # 0.2 is the weight of historical estimation, 0.8 is the weight of new data
            yt[tarea] = yt[tarea] * 0.2 + tempy[tarea] * 0.8

        # calculate the distance between data and estimation to represent data quality
        for tuser, tdata in tempdata:
            tarea = tdata[1]
            error = (tdata[3] - yt[tarea]) / yt[tarea] if yt[tarea] != 0 else tdata[3] - yt[tarea]
            error = abs(error)
            errorlist[tuser] = error
        max_error = abs(max(errorlist.items(), key=lambda x: abs(x[1]))[1])
        min_error = 0
        delta = max_error - min_error if max_error != 0 else 1
        for user in errorlist.keys():
            q = 1 - (abs(errorlist[user]) - min_error) / delta
            rep[user] = updataR(rep[user], q)
            if q > 0.5:
                de_total_e.append(telist[user])

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
    rep.clear()
    return F1, distance, 1 - ((sum(de_total_e) / len(de_total_e)) / (sum(total_e) / len(total_e)))


if __name__ == '__main__':
    version = ['origin', 'bursty', 'sparsity', 'proportion']
    type = version[0]
    if type == 'proportion':
        pp = 0.1
        path = '../Result/origin_pp/pp=' + str(pp)
    elif type == 'sparsity':
        prop = 0.6
        path = '../Result/sparsity/prop=' + str(prop)
    else:
        miu = 0.15
        sigma = 0.1
        path = '../Result/' + type + '/miu=' + str(miu) + '_sigma=' + str(sigma)
    exp_start = 1
    exp_stop = 5
    col1 = []
    col2 = []
    col3 = []
    for ver in range(exp_start, exp_stop + 1):
        init_rep = {}
        for key in range(1, 101):
            init_rep[key] = 0.5
        rpath = path + '/simulate_data/experiment' + str(ver) + '/'
        F1, distance, ate = WEI(rpath, init_rep)
        col1.append(F1)
        col2.append(distance)
        col3.append(ate)
    print(np.mean(col1))
    print(np.mean(col2))
    print(np.mean(col3))
    outpath = path + '/result.csv'
    if not os.path.exists(outpath):
        label = [
            'PRBTD F1', 'PRBTD rep dis', 'PRBTD denoise(pp)',
            'DTI F1', 'DTI rep dis', 'DTI denoise(pp)',
            'TD F1', 'TD rep dis', 'TD denoise(pp)',
            'CNB F1', 'CNB rep dis', 'CNB denoise(pp)',
            'WEI F1', 'WEI rep dis', 'WEI denoise(pp)',
        ]
        result = pd.DataFrame(columns=label)
    else:
        result = pd.read_csv(outpath)
    new_length = len(col1)
    current_length = len(result)
    if new_length != current_length:
        result = result.reindex(range(new_length))
    result['WEI F1'] = col1
    result['WEI rep dis'] = col2
    result['WEI denoise(pp)'] = col3
    result.to_csv(outpath, index=False, encoding='utf-8-sig')
