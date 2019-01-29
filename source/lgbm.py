from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
import lightgbm as lgb
import pandas as pd
import numpy as np

train = pd.read_csv('../data/processed_train_2500Hz.csv')
train = train.drop('signal_id', axis=1)

y = train['target']
X = train.drop('target', axis=1)


folds = StratifiedKFold(n_splits=5, 
                        shuffle=True, 
                        random_state=5000)

oof_preds = np.zeros(len(X))

for fold_, (trn_, val_) in enumerate(folds.split(X, y)):
    trn_x, trn_y = X.iloc[trn_], y.iloc[trn_]
    val_x, val_y = X.iloc[val_], y.iloc[val_]

    params = {'objective': 'binary',
              'seed': 5000,
              'learning_rate': 0.2,
              'num_boosting_rounds': 50,
              }
    
    clf = lgb.LGBMClassifier(**params)
    clf.fit(trn_x, trn_y, eval_set=(val_x, val_y), early_stopping_rounds=100, verbose=10)
    oof_preds[val_] = clf.predict(val_x)
    
    print('no {}-fold MCC: {}'.format(fold_ + 1, matthews_corrcoef(val_y.values, oof_preds[val_])))
    

score = matthews_corrcoef(y, oof_preds)
print('OVERALL MCC: {:.5f}'.format(score))



print(clf.feature_importances_)

print(X.head())

