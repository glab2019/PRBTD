import os.path

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import h5py
import pandas as pd


def bpmf_model(num_users, num_items, num_factors, observed_data, observed_mask):
    U = tf.Variable(tf.random.normal([num_users, num_factors]))
    V = tf.Variable(tf.random.normal([num_items, num_factors]))

    def log_likelihood():
        R_pred = tf.matmul(U, V, transpose_b=True)
        R_obs = tf.boolean_mask(R_pred, observed_mask)
        observed_data_flat = tf.boolean_mask(observed_data, observed_mask)
        return tf.reduce_sum(tfp.distributions.Normal(loc=R_obs, scale=1.0).log_prob(observed_data_flat))

    return log_likelihood, U, V


def train_bpmf(sparse_data, num_factors=10, num_steps=1000, learning_rate=0.01):
    num_users, num_items = sparse_data.shape
    observed_mask = ~np.isnan(sparse_data)
    observed_data = tf.constant(np.nan_to_num(sparse_data, nan=0.0), dtype=tf.float32)

    log_likelihood, U, V = bpmf_model(num_users, num_items, num_factors, observed_data, observed_mask)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    for step in range(num_steps):
        with tf.GradientTape() as tape:
            loss = -log_likelihood()
        grads = tape.gradient(loss, [U, V])
        optimizer.apply_gradients(zip(grads, [U, V]))
        if step % 100 == 0:
            print(f'Step {step}, Loss: {loss.numpy()}')

    filled_matrix = tf.matmul(U, V, transpose_b=True).numpy()
    return filled_matrix


def fill_matrix(input_file, output_file, data_key):
    with h5py.File(input_file, 'r') as f:
        sparse_data = f[data_key][:, :, 0]

    filled_matrix = train_bpmf(sparse_data)

    # 保持原有值，并在缺失值位置使用补全的值，同时对矩阵值进行取整
    result_matrix = np.where(np.isnan(sparse_data), filled_matrix, sparse_data)
    result_matrix = np.round(result_matrix)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset(data_key, data=result_matrix)
    print(f"Completed matrix saved to {output_file}")


if __name__ == "__main__":
    Miu = 0.3
    Sigma = 0.1

    start = 1
    end = 6
    for exp in range(start, end + 1):
        file_path = '../Result/origin/miu=' + str(Miu) + '_sigma=' + str(Sigma) + '/simulate_data/experiment' + str(
            exp) + '/'
        input_file = file_path + 'BJ16_In_mask.h5'
        output_file = file_path + 'BJ16_In_BPMF.h5'
        data_key = 'data'
        if not os.path.exists(output_file):
            fill_matrix(input_file, output_file, data_key)
        hdf = h5py.File(output_file, 'r')
        read = hdf['data'][-120:, :]

        data = pd.read_csv(file_path + 'data.csv')
        col_name = data.columns.tolist()
        index = col_name.index('prediction') + 1
        col_name.insert(index, 'BPMF')
        data = data.reindex(columns=col_name)
        bpmf = []
        for index, row in data.iterrows():
            t = int(row['time slot'])
            r = int(row['region'])
            bpmf.append(read[t][r])
        data['BPMF'] = bpmf
        data.to_csv(file_path + 'data.csv', index=False, encoding='utf-8-sig')
