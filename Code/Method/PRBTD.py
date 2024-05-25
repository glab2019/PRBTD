import csv
import math
import os.path

import numpy as np
import pandas as pd


# hyperparameters
alpha = 0.018
beta = 0.06
gamma = 0.5
EPS = 0.01
rou = 0.002


# reputation update function
def updataR(R, q):
    if q >= gamma:
        k1 = alpha
        k2 = 1 / (1 - gamma)
    else:
        k1 = beta
        k2 = 1 / gamma
    return 0.5 + (k1 * (q - gamma) * k2 + (1 - k1) * (2 * R - 1)) * 0.5


# function for calculating the environmental error
def getError(buffer):
    edict = {}
    # print(edict)
    for tbuffer in buffer:
        for data in tbuffer:
            # print(data)
            area = data[1]
            edict.setdefault(area, [0, 0])
            edict[area][0] *= edict[area][1]
            if data[3] <= 0.001:
                edict[area][0] += data[2] - data[3]
            else:
                edict[area][0] += (data[2] - data[3]) / data[3]
            edict[area][1] += 1
            edict[area][0] /= edict[area][1]
    return edict


# calculate implication between data_i and data_j
def imp(data_i, data_j, error):
    area_i = data_i[1]
    area_j = data_j[1]
    ae_i = error[area_i][0]
    ae_i = max(ae_i, 0.001) if ae_i >= 0 else min(ae_i, -0.001)
    ae_j = error[area_j][0]
    ae_j = max(ae_j, 0.001) if ae_j >= 0 else min(ae_j, -0.001)
    ei = (data_i[2] - data_i[3]) / (data_i[3]) if data_i[3] != 0 else 0
    ej = (data_j[2] - data_j[3]) / (data_j[3]) if data_j[3] != 0 else 0
    vi = np.array([ei - ae_i, ae_i])
    vj = np.array([ej - ae_j, ae_j])
    cos = vi.dot(vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
    if abs(cos) < 0.0001:
        return 0
    return cos


# Truth Discovery,
# 'list' & 'user' represent the sensing data list at this time slot and their corresponding user
# 'error' represents the environmental error dict--{region id: envir error}
# 'rep' is the reputation dict--{user id: reputation}
# 'q' is the quality dict of data in 'list'--{user id: quality of his data in this time}
# 'buff' & 'id_buff' are the data buffer and user buffer storing the sensing data that will be used
def TD(list, user, error, rep, q, buff, id_buff):
    # at the beginning of every epoch, q=rep
    for id in user:
        q[id] = rep[id]
    flag = False
    miter = 0
    # max iter = 30
    while miter < 30:
        max_c = 0
        for id in user:
            old_r = rep[id]
            rep[id] = updataR(rep[id], q[id])
            max_c = max(max_c, abs(rep[id] - old_r))
        for idx, id in enumerate(user):
            # we use rep instead of q in calculation,
            # because there is no accumulative multiplication in such situation,
            # thus according to (17) in our article, Q' = R
            theta = -math.log(1 - rep[id])
            for i, tbuff in enumerate(buff):
                for j, data in enumerate(tbuff):
                    id_j = id_buff[i][j]
                    if id_j == id:
                        continue
                    impv = imp(list[idx], data, error)
                    theta += rou * -math.log(1 - q[id_j]) * impv
            q[id] = 1 - math.exp(-theta)
            if q[id] < 0:
                q[id] = 0
        if max_c <= 0.0001 and flag:
            # when the change of r < epsilon, break the circulation
            break
        flag = True
        miter += 1
    return rep, q


def PRBTD(path, rep):
    list = {}  # store the data, {user id: data}
    q = {}  # store the quality of data and update at every new time slot,
    # {user id: quality of his data at this time slot}
    buff = []  # D in our article, the data buffer
    id_buff = []  # the user id corresponding to data in 'buff'
    buff2 = []  # the buffer storing data for calculating environmental error in every region
    te = 0  # remaining total absolute noise
    n_te = 0  # remaining data count
    totalT = 0  # the total lasting time
    with open(path + 'data.csv', encoding='utf-8-sig') as f:
        for i, data in enumerate(csv.reader(f)):
            if i == 0:
                continue
            totalT = max(totalT, int(data[1]))
            list.setdefault(int(data[0]), [])
            list[int(data[0])].append([int(data[1]), int(data[2]), float(data[4]), float(data[5]), float(data[3])])
            # list[user id]=[time slot，region，sensing value，prediction，ground truth]
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
        templist = []
        tempkey = []
        telist = {}
        flag = False
        for user in list:
            for data in list[user]:
                data_time = data[0]
                # data upload
                if time == data_time:
                    # data will be added to the list only when it is at this time slot
                    templist.append(data)  # 'templist' stores the data upload at this time slot
                    tempkey.append(user)  # the corresponding user id of data in 'templist'
                    telist[user] = abs(data[2] - data[4])  # the absolute noise of this data
                    flag = True

        buff.append(templist)
        id_buff.append(tempkey)
        buff2.append(templist)

        if len(buff) > 5:  # we use the data in last 5 time slot to calculate quality
            buff.pop(0)
            id_buff.pop(0)
        if len(buff2) > 100:  # we use the data in last 100 time slot to calculate environmental error
            buff2.pop(0)
        error = getError(buff2)
        if flag:
            rep, q = TD(templist, tempkey, error, rep, q, buff, id_buff)
            for id in tempkey:
                if q[id] > 0.5:
                    te += telist[id]
                    n_te += 1

    slist = sorted(rep.items(), key=lambda x: x[1], reverse=False)
    wrong = 0
    for i in range(len(b_worker)):
        if slist[i][0] not in b_worker:
            # print(slist[i])
            wrong += 1
    bad = 0
    good = 0
    for key in rep.keys():
        if key in b_worker:
            bad += rep[key]
        else:
            good += rep[key]
    distance = good / (len(list) - len(b_worker)) - bad / len(b_worker)
    acc = (len(b_worker) - wrong) / len(b_worker)
    list.clear()
    rep.clear()
    q.clear()
    buff.clear()
    id_buff.clear()
    buff2.clear()
    # print('wrong choice:', wrong)
    # print('identification acc:', acc)
    # print('avr of good - avr of bad:', distance)
    # print('remaing noise:', te / n_te)
    return acc, distance, te / n_te


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
        acc, distance, ate = PRBTD(rpath, init_rep)
        col1.append(acc)
        col2.append(distance)
        col3.append(ate)
    print(np.mean(col1))
    print(np.mean(col2))
    print(np.mean(col3))
    outpath = path + '/result.csv'
    result = pd.read_csv(outpath)
    result['PRBTD acc'] = col1
    result['PRBTD rep dis'] = col2
    result['PRBTD noise'] = col3
    result.to_csv(outpath, index=False, encoding='utf-8-sig')
