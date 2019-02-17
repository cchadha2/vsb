# Note: Could use SMOTE to handle undersampling negative samples while
# oversampling positive samples

import pandas as pd

train = pd.read_csv('../data/metadata_train.csv')
train_sig = pd.read_parquet('../data/train.parquet')
SEED = 1001

positive_samples = train[train['target'] == 1]

print(positive_samples.shape)

num_negative_samples = positive_samples.shape[0]
negative_samples = train[train['target'] == 0].sample(
    num_negative_samples, random_state=SEED)

print(negative_samples.shape)

train_sampled = pd.concat([positive_samples, negative_samples])

print(train_sampled.shape)

train_sampled.to_csv('../output/train_sampled.csv')

samples = train_sampled['signal_id'].values.tolist()
samples = [str(x) for x in samples]

print(samples)

train_sig_sampled = train_sig[samples]

train_sig_sampled.to_parquet('../output/train_sig_sampled.parquet')
