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


def comparison_TD(path, rep):
    list = {}
    pred = []
    yt = {}
    totalT = 0
    total_e = []
    de_total_e = []
    with open(path + 'data.csv', encoding='utf-8-sig') as f:
        for i, data in enumerate(csv.reader(f)):
            if i == 0:
                continue
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
    for time in range(totalT + 1):
        errorlist = {}
        tempdata = []
        tempy = {}
        tempcnt = {}
        telist = {}
        storelist = {}  # key=area, item=[user in this area at this time slot]
        for user in list.keys():
            for data in list[user]:
                # data will be added to the list only when it is in this time slot
                if time == data[0]:
                    area = data[1]
                    tempy.setdefault(area, 0)
                    tempy[area] += data[3]
                    tempcnt.setdefault(area, 0)
                    tempcnt[area] += 1
                    tempdata.append([user, data])
                    telist[user] = abs(data[3] - data[2])
                    storelist.setdefault(area, [])
                    storelist[area].append([user, data])
                    total_e.append(abs(data[3] - data[2]))
        # update the errors of areas
        for tarea in tempy.keys():
            yt.setdefault(tarea, tempy[tarea] / tempcnt[tarea])
            yt[tarea], rep = TD(yt[tarea], storelist[tarea], rep)

        # calculate user errors
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


def TD(oldY, ulist, rep):
    output = oldY
    iter = 0
    while iter < 50:
        ttweight = 0
        ysum = 0
        errorlist = {}
        for user, data in ulist:
            ttweight += rep[user]
        for user, data in ulist:
            ysum += rep[user] / ttweight * data[3]
        # estimation is based on both the historical estimation and new data
        newy = ysum * 0.8 + oldY * 0.2
        iter += 1

        for user, data in ulist:

            error = (data[3] - newy) / newy if newy != 0 else data[3] - newy
            error = abs(error)
            if error <= 0.001: error = 0
            errorlist[user] = error
        max_error = abs(max(errorlist.items(), key=lambda x: abs(x[1]))[1])
        min_error = 0
        delta = max_error - min_error if max_error != 0 else 1
        for user in errorlist.keys():
            q = 1 - (abs(errorlist[user]) - min_error) / delta
            rep[user] = updataR(rep[user], q)
        if abs(newy - oldY) <= 0.001:
            output = newy
            break
        oldY = newy
    return output, rep


if __name__ == '__main__':
    version = ['origin','bursty','sparsity','proportion']
    type = version[0]
    if type == 'proportion':
        pp = 0.1
        path = '../Result/origin_pp/pp=' + str(pp)
    elif type == 'sparsity':
        prop = 0.9
        path = '../Result/sparsity/prop=' + str(prop)
    else:
        miu = 0.3
        sigma = 0.1
        path = '../Result/' + type + '/miu=' + str(miu) + '_sigma=' + str(sigma)
    exp_start = 1
    exp_stop = 6
    col1 = []
    col2 = []
    col3 = []
    init_rep = {}
    for ver in range(exp_start, exp_stop + 1):
        for key in range(1, 101):
            init_rep[key] = 0.5
        rpath = path + '/simulate_data/experiment' + str(ver) + '/'
        F1, distance, noise = comparison_TD(rpath, init_rep)
        col1.append(F1)
        col2.append(distance)
        col3.append(noise)
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
    result['TD F1'] = col1
    result['TD rep dis'] = col2
    result['TD denoise(pp)'] = col3
    result.to_csv(outpath, index=False, encoding='utf-8-sig')
