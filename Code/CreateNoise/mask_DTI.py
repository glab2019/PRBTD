import os.path
import csv
import pandas as pd
import numpy as np
import h5py

Miu = 0.3
Sigma = 0.1
bia = 7
file_path = '../Predict/data/'
hdf = h5py.File(file_path + 'BJ16_In.h5', 'r')
read = hdf['data'][()]
sample = [bia + 32 * i for i in range(32)]
data = read[-240:, sample, 0]

start = 1
end = 6
for exp in range(start, end + 1):
    if os.path.exists('../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/simulate_data/experiment' + str(
            exp) + '/BJ16_In_mask.h5'):
        continue
    remove = []
    df = pd.read_csv('../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/simulate_data/experiment' + str(
        exp) + '/data.csv')
    # df_noise = pd.read_csv('../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/simulate_data/experiment' + str(exp) + '/noise.csv')

    for _, row in df.iterrows():
        t = int(row['time slot'])
        r = int(row['region'])
        index_r = (t + 120) * 32 + r
        if index_r not in remove:
            remove.append(index_r)

    mask = np.ones(data.shape, dtype=bool)
    mask[np.unravel_index(remove, data.shape)] = False

    # Create a sparse matrix with missing data
    sparse_data = np.where(mask, data, np.nan)
    sparse_data = sparse_data.reshape((240, -1, 1))
    sparse_data = np.repeat(sparse_data, 3, axis=2)
    # Save the sparse data back to the h5 file
    # hdf.create_dataset('sparse_data', data=sparse_data)
    with h5py.File('../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/simulate_data/experiment' + str(
            exp) + '/BJ16_In_mask.h5', "w") as hf:
        # hf.create_group("dataset")
        i = hf.create_dataset("data", data=sparse_data, dtype='float64')
        hf.close()

    # print('remove time',remove_t)
    # print('remove region',remove_r)
