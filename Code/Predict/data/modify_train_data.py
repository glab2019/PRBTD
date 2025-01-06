import h5py
import numpy as np
import random

np.random.seed(0)
random.seed(0)
f = h5py.File('BJ16_In.h5', 'r')
read = f['data'][()]
taxi_in = read[:, :, 0]
t, n = taxi_in.shape
t = t - 120
# the proportion of training data with noise
prop = 0.1
sample_index = int(t * n * prop)
#
# indexs = random.sample(range(0, t * n), sample_index)
#
# print(len(indexs))
noise = np.zeros((t + 120, n))
indexs = np.random.choice(t * n, sample_index, replace=False)
noise.flat[indexs] = np.random.normal(0, 0.1, sample_index)
noise = np.floor(noise * taxi_in)

taxi_noise = taxi_in + noise
taxi_noise = np.where(taxi_noise < 0, 0, taxi_noise)

# print(taxi_noise[0, 52], taxi_in[0, 52])
taxi_noise = taxi_noise.reshape((7220, 32 * 32, 1))
taxi_noise = np.repeat(taxi_noise, 3, axis=2)
taxi_noise.tolist()
with h5py.File("BJ16_In_pp=" + str(prop) + ".h5", "w") as hf:
    # hf.create_group("dataset")
    i = hf.create_dataset("data", data=taxi_noise, dtype='float64')
    hf.close()
