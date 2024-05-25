import os.path
import csv
import pandas as pd
import random

seed = 16
random.seed(seed)

prop = 0.9  # proportion of data, represents the sparsity of data
Miu = 0.3
Sigma = 0.1

out = '../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma)
if not os.path.exists(out):
    os.makedirs(out)

df = pd.read_csv(
    '../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) +
    '/simulate_data' + '/experiment1/data.csv')

gdata = []
bdata = []
label = [[
    'PRBTD F1', 'PRBTD rep dis', 'PRBTD noise',
    'TD F1', 'TD rep dis', 'TD noise',
    'CNB F1', 'CNB rep dis', 'CNB noise',
    'WEI F1', 'WEI rep dis', 'WEI noise',
]]

for index, row in df.iterrows():
    if float(row['noise(pp)']) != 0:
        bdata.append(index)
    else:
        gdata.append(index)
notb = random.sample(bdata, round(len(bdata) * (1 - prop)))
notg = random.sample(gdata, round(len(gdata) * (1 - prop)))

exclude = notb + notg

start = 1
end = 1
for exp in range(start, end + 1):
    dict = {}
    df = pd.read_csv(
        '../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) +
        '/simulate_data' + '/experiment' + str(exp) + '/data.csv')
    df_noise = pd.read_csv(
        '../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) +
        '/simulate_data' + '/experiment' + str(exp) + '/noise.csv')

    result_df = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        if index in exclude:
            continue
        result_df = pd.concat([result_df, pd.DataFrame(row).T])
        if float(row['noise(pp)']) != 0:
            dict.setdefault(row['user'], 1)
            # list.setdefault(int(data[0]), [])

    outpath = '../Result/sparsity/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/simulate_data/experiment' + str(exp)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    out1 = outpath + '/data.csv'
    out2 = outpath + '/noise.csv'

    result_df['user'] = result_df['user'].astype(int)
    result_df['time slot'] = result_df['time slot'].astype(int)
    result_df['region'] = result_df['region'].astype(int)
    result_df.to_csv(out1, index=False, encoding='utf-8-sig')
    df_noise.to_csv(out2, index=False, encoding='utf-8-sig')
    print(len(dict))

with open('../Result/sparsity/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/result.csv', 'w', newline='',
          encoding='utf-8-sig') as result:
    writer = csv.writer(result)
    writer.writerows(label)
    result.close()
