import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('../data/processed_train_2500Hz_1.1.csv')
test = pd.read_csv('../data/processed_test_2500Hz_1.1.csv')

# train = train.drop('signal_id', axis=1)
train = train.drop(['signal_id',
                    # 'amplitude__first_location_of_maximum',
                    'amplitude__number_peaks__n_100',
                    # 'amplitude__range_count__max_-10__min_-15',
                    # 'amplitude__range_count__max_15__min_10',
                    'amplitude__symmetry_looking__r_1',
                    'min_peak_height'], axis=1)

sub_df = test.drop([column for column in test.columns if column not in [
                   'signal_id']], axis=1).reset_index(drop=True)
sub_df['target'] = 0
# test = test.drop(['signal_id'], axis=1)
test = test.drop(['signal_id',
                  # 'amplitude__first_location_of_maximum',
                  'amplitude__number_peaks__n_100',
                  # 'amplitude__range_count__max_-10__min_-15',
                  # 'amplitude__range_count__max_15__min_10',
                  'amplitude__symmetry_looking__r_1',
                  'min_peak_height'], axis=1)
num_folds = 10
SEED = 5000
version = '1.5'
y = train['target']
X = train.drop('target', axis=1)
features = [feature for feature in X.columns if feature not in [
    'signal_id', 'amplitude']]
X = X[features]
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
sub_preds = np.zeros(test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X, y)):
    train_x, train_y = X.iloc[trn_], y.iloc[trn_]
    val_x, val_y = X.iloc[val_], y.iloc[val_]

    params = {'num_leaves': 80,
              'min_data_in_leaf': 60,
              'objective': 'binary',
              'max_depth': -1,
              'learning_rate': 0.1,
              "boosting": "gbdt",
              "feature_fraction": 0.8,
              "bagging_freq": 1,
              "bagging_fraction": 0.8,
              "bagging_seed": 11,
              "metric": 'auc',
              "lambda_l1": 0.1,
              "random_state": SEED,
              "num_iterations": 1000,
              "verbosity": -1}

    clf = lgb.LGBMClassifier(**params)
    clf = clf.fit(train_x, train_y, eval_set=[
                  (val_x, val_y)], early_stopping_rounds=100)

    oof_preds[val_] = clf.predict(val_x)
    print(oof_preds)

    sub_preds += clf.predict(test[features].values) / num_folds
    print(sub_preds)

    print('no {}-fold MCC: {}'.format(fold_ + 1,
                                      matthews_corrcoef(y.iloc[val_].values, oof_preds[val_])))


sub_df['target'] = sub_preds
print(sub_df.head(20))
sub_df['target'] = sub_df['target'].apply(lambda x: round(x, 0))
sub_df['target'] = sub_df['target'].astype(int)
sub_df.to_csv('../preds/lgbm_preds_2500Hz_' + version + '.csv', index=False)
