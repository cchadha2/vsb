from tsfresh import extract_features
import pandas as pd

train_sig = pd.read_csv('../output/train_sig_2500Hz_4th_order_dwt_tsfresh_format.csv')

fc =   { 'binned_entropy': [{'max_bins': 10}],
        'fft_coefficient': [{'coeff': 0, 'attr': 'abs'}, {'coeff': 1, 'attr': 'abs'}],
        'value_count': [{'value': 10}, {'value': 8}],
        'count_above_mean': None,
        'count_below_mean': None,
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean': None,
        'mean_abs_change': None,
        'mean_change': None,
        'median': None }

train_sig_with_features = extract_features(train_sig,
                                           column_id='signal_id',
                                           column_value='amplitude',
                                           default_fc_parameters=fc)


print(train_sig_with_features.head())

train_sig_with_features.to_csv('../output/train_sig_2500Hz_with_ts_features.csv', index=False)

train_meta = pd.read_csv('../output/train_sampled.csv')
train = pd.concat([train_sig_with_features, train_meta], axis=1)

train.to_csv('../data/processed_train_2500Hz.csv', index=False)