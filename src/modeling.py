import numpy as np
import pandas as pd
from pymongo import MongoClient

import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import f1_score

client = MongoClient('localhost', 27016)
db = client['foia_requests']
foia_data = db['foia_jsons']

foias = list(foia_data.find({'body' : {"$exists": True}}))

df = pd.read_csv("../data/agency_ids.csv")
df.set_index('id', inplace=True)
df = df.append(pd.DataFrame(data=[['Agency not found']],
                       columns=df.columns,
                       index=[0]))

agencies = pd.Series(index=df.index, data=df['name'])

df['count'] = df.index * 0
statuses = ['rejected', 'fix', 'no_docs', 'partial', 'done']

for status in statuses:
    df[status] = df['count']

for foia in foias:
    try:
        df.loc[foia['agency'], 'count'] += 1
        df.loc[foia['agency'], foia['status']] += 1
    except KeyError:
        df.loc[0, 'count'] += 1
        df.loc[0, foia['status']] += 1

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

corpus = [foia['body'] for foia in foias]

tfidf_matrix = vectorizer.fit_transform(corpus)

idx_to_word = {vectorizer.vocabulary_[word] : word
               for word in vectorizer.vocabulary_}

X = pd.DataFrame(data=tfidf_matrix.toarray(),
                 columns=[idx_to_word[i] for i in range(5000)])

X['request_agency'] = [foia['agency'] for foia in foias]

y = np.array([foia['status'] for foia in foias])

X.to_csv('../data/model_features.csv')
pd.Series(data=y).to_csv('../data/target.csv')

lgbclf = lgb.LGBMClassifier()
lrclf = LogisticRegression()
knnclf = KNeighborsClassifier()
rfclf = RandomForestClassifier()

kf = KFold(n_splits=5, shuffle=True)

lgbtrain_fold_scores = []
lrtrain_fold_scores = []
knntrain_fold_scores = []
rftrain_fold_scores = []
lgbtest_fold_scores = []
lrtest_fold_scores = []
knntest_fold_scores = []
rftest_fold_scores = []

for train, test in kf.split(X):
    lgbclf.fit(X.values[train], y[train])
    lrclf.fit(X.values[train], y[train])
    knnclf.fit(X.values[train], y[train])
    rfclf.fit(X.values[train], y[train])
    lgbtrain_fold_scores.append(f1_score(y[train], lgbclf.predict(X.values[train]), average='weighted'))
    lrtrain_fold_scores.append(f1_score(y[train], lrclf.predict(X.values[train]), average='weighted'))
    knntrain_fold_scores.append(f1_score(y[train], knnclf.predict(X.values[train]), average='weighted'))
    rftrain_fold_scores.append(f1_score(y[train], rfclf.predict(X.values[train]), average='weighted'))
    lgbtest_fold_scores.append(f1_score(y[test], lgbclf.predict(X.values[test]), average='weighted'))
    lrtest_fold_scores.append(f1_score(y[test], lrclf.predict(X.values[test]), average='weighted'))
    knntest_fold_scores.append(f1_score(y[test], knnclf.predict(X.values[test]), average='weighted'))
    rftest_fold_scores.append(f1_score(y[test], rfclf.predict(X.values[test]), average='weighted'))

print("lgb train score:", np.mean(lgbtrain_fold_scores))
print("lr train score:", np.mean(lrtrain_fold_scores))
print("knn train score:", np.mean(knntrain_fold_scores))
print("rf train score:", np.mean(rftrain_fold_scores))
print("lgb test score:", np.mean(lgbtest_fold_scores))
print("lr test score:", np.mean(lrtest_fold_scores))
print("knn test score:", np.mean(knntest_fold_scores))
print("rf test score:", np.mean(rftest_fold_scores))