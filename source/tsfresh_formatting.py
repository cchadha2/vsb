import pandas as pd

train_sig = pd.read_parquet('../output/train_sig_2500hz_4th_order_dwt.parquet')
train_meta = pd.read_csv('../output/train_sampled.csv', index_col=0)

amplitude = pd.DataFrame(
    train_sig.stack().reset_index().drop(
        'level_0', axis=1).rename(
            columns={
                'level_1': 'signal_id'}))
amplitude['signal_id'] = amplitude['signal_id'].astype('int')

signal_id = pd.DataFrame(train_meta['signal_id']).reset_index(drop=True)
tsfresh_df = pd.merge(
    signal_id,
    amplitude,
    how='inner').rename(
        columns={
            0: 'amplitude'})

print(tsfresh_df.head())

tsfresh_df.to_csv(
    '../output/train_sig_2500Hz_4th_order_dwt_tsfresh_format.csv',
    index=False)
