import os.path
import csv
import pandas as pd
import random

seed = 66
random.seed(seed)

Miu = 0.3
Sigma = 0.1
delta_t = 6  # affect data in delta_t+1 time slots
delta_r = 3  # affect data in delta_t+1 regions

inc_r = []
for i in range(3):
    start_t = random.randint(1, 120 - delta_t)
    start_r = random.randint(0, 32 - delta_r)
    print('start time:', start_t, 'start region:', start_r)
    inc_r.append([start_t, start_r])

out = '../Result/bursty/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/simulate_data/'
if not os.path.exists(out):
    os.makedirs(out)

start = 1
end = 1
for exp in range(start, end + 1):
    outpath = out + '/experiment' + str(exp)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    out1 = outpath + '/data.csv'
    out2 = outpath + '/noise.csv'
    df = pd.read_csv('../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/simulate_data/experiment' + str(
        exp) + '/data.csv')
    df_noise = pd.read_csv('../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/simulate_data/experiment' + str(exp) + '/noise.csv')
    col_name = df.columns.tolist()
    index = col_name.index('noise(pp)') + 1
    col_name.insert(index, 'with burst')
    df = df.reindex(columns=col_name)
    df['with burst'] = [int(0)] * df.shape[0]

    result_df = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():
        t = row['time slot']
        r = row['region']
        flag = 0
        for st, sr in inc_r:
            if st <= t <= st + delta_t and sr <= r <= sr + delta_r:
                flag = 1
        if flag:
            old = row['ground truth']
            row['ground truth'] = round(row['ground truth'] / 2)
            # we do not use row['sensing traffic'] = round(row['ground truth'] * (1 + row['noise(pp)']))
            # as it will reduce the total noise of data, which will affect the metric 3 in our article
            row['sensing traffic'] = row['sensing traffic'] + row['ground truth'] - old
            row['with burst'] = 1
        result_df = pd.concat([result_df, pd.DataFrame(row).T])

    result_df['user'] = result_df['user'].astype(int)
    result_df['time slot'] = result_df['time slot'].astype(int)
    result_df['region'] = result_df['region'].astype(int)
    result_df.to_csv(out1, index=False, encoding='utf-8-sig')
    df_noise.to_csv(out2, index=False, encoding='utf-8-sig')

