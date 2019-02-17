import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('../data/processed_train_2500Hz_1.2.csv')
train = train.drop(['signal_id'], axis=1)

num_folds = 10
SEED = 5000
y = train['target']
X = train.drop('target', axis=1)
features = [feature for feature in X.columns if feature not in ['signal_id', 'amplitude']]
X = X[features]
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
version = '1.2_ranged'

oof_preds = np.zeros(len(X))
feature_importance_df = pd.DataFrame()

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

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = X.columns.tolist()
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat(
        [feature_importance_df, fold_importance_df], axis=0)

    print('no {}-fold MCC: {}'.format(fold_ + 1,
                                      matthews_corrcoef(y.iloc[val_].values, oof_preds[val_])))

score = matthews_corrcoef(y, oof_preds)
print('OVERALL MCC: {:.5f}'.format(score))

feature_importance_df.groupby('feature', as_index=False).mean().drop('fold', axis=1)\
    .sort_values('importance', ascending=False).to_csv('../output/lgbm_importance_' + version + '.csv', index=False)
