import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.signal import butter, filtfilt

SEED = 5000
random.seed(SEED)
train_sig = pd.read_parquet('../output/train_sig_sampled.parquet')

print(train_sig.head())
print(train_sig.shape)

cutoff = 10000
measurements = 800000
time = 0.02
sampling_rate = measurements / time
nyquist = sampling_rate * 0.5
wn = cutoff / nyquist

b, a = butter(5, wn, btype='highpass')

filtered_sig = filtfilt(b, a, train_sig.values, axis=0)
train_sig_filtered = pd.DataFrame(
    data=filtered_sig,
    columns=train_sig.columns.tolist())

print(train_sig_filtered.head())
print(train_sig_filtered.shape)

signals_to_plot = random.choices(train_sig_filtered.columns.tolist(), k=3)
for i in signals_to_plot:
    plt.figure()
    sns.lineplot(train_sig_filtered[:800000].index,
                 train_sig_filtered[str(i)][:800000])
    plt.show()


train_sig_filtered.to_parquet('../output/train_sig_10000hz_5th_order.parquet')
