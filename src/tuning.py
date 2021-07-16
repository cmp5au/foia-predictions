import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
from pymongo import MongoClient

X = pd.read_csv('../data/model_features.csv')
y = pd.read_csv('../data/target.csv').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

categoricals = ['request_agency']
indexes_of_categories = [X.columns.get_loc(col) for col in categoricals]

gkf = KFold(n_splits=5, shuffle=True).split(X_train)

param_grid = {
    'num_leaves': [31, 127],
    'objective': ['multiclass', 'crossentropy'],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }

lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                                   num_boost_round=2000,
                                   learning_rate=0.01,
                                   metric='multi_logloss')

gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=gkf)
lgb_model = gsearch.fit(X=X_train, y=y_train, categorical_feature=indexes_of_categories)

print("Grid search output:", lgb_model.best_params_, lgb_model.best_score_)

toplgbclf = lgb.LGBMClassifier(boosting_type='gbdt',
                               num_boost_round=2000,
                               learning_rate=0.01,
                               metric='multi_logloss',
                               **lgb_model.best_params_)

toplgbclf.fit(X_train, y_train)
print("Overall F1 score:",
      f1_score(y_test, toplgbclf.predict(X_test.values), average='weighted'))
print("Overall accuracy:", (y_test == toplgbclf.predict(X_test.values)).mean())

fig, ax = plt.subplots()

lgb.plot_importance(toplgbclf, ax=ax, max_num_features=10, importance_type='gain')
fig.tight_layout()
plt.savefig('../images/feature_importances_gain.png')