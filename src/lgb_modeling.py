import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import lightgbm as lgb
import re

df = pd.read_csv('../data/body_multiclass_target.csv')
df['request_agency'] = pd.read_csv('../data/model_features.csv')['request_agency']
df.dropna(axis=0, inplace=True)

new_labels = {'done' : 'Completed',
              'partial' : 'Completed',
              'no_docs' : 'Redacted',
              'fix' : 'Rejected',
              'rejected' : 'Rejected'}

df['target'] = df.target.apply(lambda x: new_labels[x])

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
corpus = df['body']

train, test = train_test_split(np.arange(len(df)), test_size=0.2,
                               random_state=42, stratify=df.target)

tfidf.fit(corpus.values[train])
X_1 = tfidf.transform(corpus.values).toarray()
X_2 = df['request_agency'].values.reshape(-1, 1)
print("Shapes of X_1 and X_2:", X_1.shape, X_2.shape)
X = np.concatenate((X_1, X_2), axis=1)
y = df['target'].values

# this next line is only to remove bits of tokens that are problematic
# for a LightGBM model
# X = X.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
# target_df = pd.read_csv('../data/target.csv')

# new_labels = {'done' : 'Completed',
#               'partial' : 'Completed',
#               'no_docs' : 'Redacted',
#               'fix' : 'Rejected',
#               'rejected' : 'Rejected'}

# target_df['target'] = target_df['0'].apply(lambda x: new_labels[x])

# y = target_df['0'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

colors = np.array(['g' if y == 'Completed' else 'k' if y == 'Redacted' else 'r'
                    for y in y[test]])

categoricals = ['request_agency']
indexes_of_categories = [X.shape[1] - 1]

lgb_model = lgb.LGBMClassifier(num_leaves=127,
                                   boosting_type='gbdt',
                                   objective='crossentropy',
                                   reg_alpha=0.1,
                                   num_boost_round=2000,
                                   learning_rate=0.01,
                                   metric='multi_logloss')
lgb_model.fit(X=X[train], y=y[train], categorical_feature=indexes_of_categories)

preds = lgb_model.predict(X[test])
proba_preds = lgb_model.predict_proba(X[test])

print("Overall F1 score:",
      f1_score(y[test], preds, average='weighted'))
print("Overall accuracy:", (y[test] == preds).mean())

# fig, ax = plt.subplots()

# lgb.plot_importance(toplgbclf, ax=ax, max_num_features=10, importance_type='gain')
# fig.tight_layout()
# plt.savefig('../images/feature_importances_gain.png')

fig, ax = plt.subplots()

tsne = TSNE(n_components=2)

np.random.seed(42)
idxs = np.random.choice(range(X[test].shape[0]), size=1000, replace=False)

scatter = tsne.fit_transform(proba_preds[idxs])
xs = scatter[:, 0]
ys = scatter[:, 1]
# zs = scatter[:, 2]

ax.scatter(xs, ys, c=colors[idxs])
ax.set_title("Testing LightGBM Classifier output with TSNE")
plt.savefig('../images/lgbm_tsne_2d.png')

tsne_3d = TSNE(n_components=3)
scatter_3d = tsne_3d.fit_transform(proba_preds[idxs])
xs_3d = scatter_3d[:, 0]
ys_3d = scatter_3d[:, 1]
zs_3d = scatter_3d[:, 2]

ax = plt.figure().gca(projection='3d')

ax.scatter(xs_3d, ys_3d, zs_3d, c=colors[idxs])
ax.set_title("Testing LightGBM Classifier output with TSNE in 3D")
plt.savefig('../images/lgbm_tsne_3d.png')