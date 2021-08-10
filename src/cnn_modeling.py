import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from convnets_utils import *
from plot_tsne import plot_tsne

df = pd.read_csv('../data/body_multiclass_target.csv')
df['agency'] = pd.read_csv('../data/model_features.csv')['request_agency']
new_labels = {'done' : 'Completed',
              'partial' : 'Completed',
              'no_docs' : 'Redacted',
              'fix' : 'Rejected',
              'rejected' : 'Rejected'}
df.loc[:, 'target'] = df.target.apply(lambda x: new_labels[x])
df.dropna(axis=0, inplace=True)

train, test = train_test_split(np.arange(len(df)),
                                test_size=0.15,
                                random_state=42,
                                stratify=df.target.values)

X = df.body.values
y = df.target.values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X[train])
joblib.dump(tokenizer, '../models/tokenizer')
sequences_train = tokenizer.texts_to_sequences(X[train])
sequences_test = tokenizer.texts_to_sequences(X[test])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# truncate or pad all the articles to the same length
sequences_train = [x[:250] for x in sequences_train]
sequences_test = [x[:250] for x in sequences_test]
data_train = pad_sequences(sequences_train, maxlen=250, padding='post', truncating='post')
data_test = pad_sequences(sequences_test, maxlen=250, padding='post', truncating='post')

le = LabelEncoder()
le.fit(y[train])
labels_encoded_train = le.transform(y[train])
labels_encoded_test = le.transform(y[test])
categorical_labels_train = to_categorical(labels_encoded_train, num_classes=None)
categorical_labels_test = to_categorical(labels_encoded_test, num_classes=None)
print('Shape of train data tensor:', data_train.shape)
print('Shape of train label tensor:', categorical_labels_train.shape)

model_1 = get_cnn_rand(300, len(word_index)+1, 250, 3, loss='binary_crossentropy')
history_1 = model_1.fit(x=data_train, y=categorical_labels_train, validation_split=0.2, batch_size=50, epochs=3)

raw_predictions_1 = model_1.predict(data_test)
class_predictions_1 = [np.argmax(x) for x in raw_predictions_1]
print("Results for CNN with random word embeddings:")
print(classification_report(y[test], le.inverse_transform(class_predictions_1)))

# model_1.save('../models/model_1')
# joblib.dump(model_1, '../models/model_1.jl')
fig, ax = plt.subplots()
plot_tsne(ax, model_1, data_test, y[test], title='Output of CNN with random word embeddings', raw_preds=raw_predictions_1)
plt.savefig('../images/tSNE/2D/cnn_1_tsne_2d.png')

embeddings_index = load_fasttext_embeddings()
embeddings_matrix = create_embeddings_matrix(embeddings_index, word_index, 100)
embedding_layer_static = get_embeddings_layer(embeddings_matrix, 'embedding_layer_static', 250, trainable=False)
model_2 = get_cnn_pre_trained_embeddings(embedding_layer_static, 250, 3)

history_2 = model_2.fit(x=data_train, y=categorical_labels_train, batch_size=50, epochs=10)

raw_predictions_2 = model_2.predict(data_test)
class_predictions_2 = [np.argmax(x) for x in raw_predictions_2]
print("Results for CNN with pre-trained static word embeddings")
print(classification_report(y[test], le.inverse_transform(class_predictions_2)))

# model_2.save('../models/model_2')
# joblib.dump(model_2, '../models/model_2.jl')
fig, ax = plt.subplots()
plot_tsne(ax, model_2, data_test, y[test], title='Output of CNN with pre-trained static word embeddings', raw_preds=raw_predictions_2)
plt.savefig('../images/tSNE/2D/cnn_2_tsne_2d.png')

embedding_layer_dynamic = get_embeddings_layer(embeddings_matrix, 'embedding_layer_dynamic', 250, trainable=True)
model_3 = get_cnn_pre_trained_embeddings(embedding_layer_dynamic, 250, 3)

history_3 = model_3.fit(x=data_train, y=categorical_labels_train, batch_size=50, epochs=10)

raw_predictions_3 = model_3.predict(data_test)
class_predictions_3 = [np.argmax(x) for x in raw_predictions_3]
print("Results for CNN with pre-trained dynamic word embeddings")
print(classification_report(y[test], le.inverse_transform(class_predictions_3)))

# model_3.save('../models/model_3')
# joblib.dump(model_3, '../models/model_3.jl')
fig, ax = plt.subplots()
plot_tsne(ax, model_3, data_test, y[test], title='Output of CNN with pre-trained dynamic word embeddings', raw_preds=raw_predictions_3)
plt.savefig('../images/tSNE/2D/cnn_3_tsne_2d.png')

model_4 = get_cnn_multichannel(embedding_layer_static, embedding_layer_dynamic, 250, 3)

history_4 = model_4.fit(x=[data_train, data_train], y=categorical_labels_train, batch_size=50, epochs=10)

raw_predictions_4 = model_4.predict([data_test, data_test])
class_predictions_4 = [np.argmax(x) for x in raw_predictions_4]
print("Results for multichannel CNN with pre-trained dynamic and static word embeddings")
print(classification_report(y[test], le.inverse_transform(class_predictions_4)))

# model_4.save('../models/model_4')
# joblib.dump(model_4, '../models/model_4.jl')
fig, ax = plt.subplots()
plot_tsne(ax, model_4, [data_test, data_test], y[test], title='CNN with pre-trained dynamic and static word embeddings', raw_preds=raw_predictions_4)
plt.savefig('../images/tSNE/2D/cnn_4_tsne_2d.png')

import lightgbm as lgb

lgbclf_1 = lgb.LGBMClassifier(categorical_feature=5,
                              boosting_type='gbdt',
                              num_boost_round=2000,
                              learning_rate=0.01,
                              metric='multi_logloss')

W_1 = np.concatenate((model_1.predict(data_train), df.agency.values[train].reshape(-1, 1)), axis=1)
Z_1 = np.concatenate((model_1.predict(data_test), df.agency.values[test].reshape(-1, 1)), axis=1)
lgbclf_1.fit(W_1, y[train])
# joblib.dump(lgbclf_1, '../models/lgbclf_1')
raw_lgbm_predictions_1 = lgbclf_1.predict_proba(Z_1)
class_lgbm_predictions_1 = [np.argmax(x) for x in raw_lgbm_predictions_1]
print("Results for LGBM Classifier on top of Model #1")
print(classification_report(y[test], le.inverse_transform(class_lgbm_predictions_1)))
fig, ax = plt.subplots()
plot_tsne(ax, lgbclf_1, Z_1, y[test], title='LightGBM on CNN with random word embeddings', raw_preds=raw_lgbm_predictions_1)
plt.savefig('../images/tSNE/2D/lgbm_cnn_1_tsne_2d.png')

lgbclf_2 = lgb.LGBMClassifier(categorical_feature=5,
                              boosting_type='gbdt',
                              num_boost_round=2000,
                              learning_rate=0.01,
                              metric='multi_logloss')

W_2 = np.concatenate((model_2.predict(data_train), df.agency.values[train].reshape(-1, 1)), axis=1)
Z_2 = np.concatenate((model_2.predict(data_test), df.agency.values[test].reshape(-1, 1)), axis=1)
lgbclf_2.fit(W_2, y[train])
# joblib.dump(lgbclf_2, '../models/lgbclf_2')
raw_lgbm_predictions_2 = lgbclf_2.predict_proba(Z_2)
class_lgbm_predictions_2 = [np.argmax(x) for x in raw_lgbm_predictions_2]
print("Results for LGBM Classifier on top of Model #2")
print(classification_report(y[test], le.inverse_transform(class_lgbm_predictions_2)))
fig, ax = plt.subplots()
plot_tsne(ax, lgbclf_2, Z_2, y[test], title='LightGBM on CNN with static word embeddings', raw_preds=raw_lgbm_predictions_2)
plt.savefig('../images/tSNE/2D/lgbm_cnn_2_tsne_2d.png')

lgbclf_3 = lgb.LGBMClassifier(categorical_feature=5,
                              boosting_type='gbdt',
                              num_boost_round=2000,
                              learning_rate=0.01,
                              metric='multi_logloss')

W_3 = np.concatenate((model_3.predict(data_train), df.agency.values[train].reshape(-1, 1)), axis=1)
Z_3 = np.concatenate((model_3.predict(data_test), df.agency.values[test].reshape(-1, 1)), axis=1)
lgbclf_3.fit(W_3, y[train])
# joblib.dump(lgbclf_3, '../models/lgbclf_3')
raw_lgbm_predictions_3 = lgbclf_3.predict_proba(Z_3)
class_lgbm_predictions_3 = [np.argmax(x) for x in raw_lgbm_predictions_3]
print("Results for LGBM Classifier on top of Model #3")
print(classification_report(y[test], le.inverse_transform(class_lgbm_predictions_3)))
fig, ax = plt.subplots()
plot_tsne(ax, lgbclf_3, Z_3, y[test], title='LightGBM on CNN with dynamic word embeddings', raw_preds=raw_lgbm_predictions_3)
plt.savefig('../images/tSNE/2D/lgbm_cnn_3_tsne_2d.png')

lgbclf_4 = lgb.LGBMClassifier(categorical_feature=5,
                              boosting_type='gbdt',
                              num_boost_round=2000,
                              learning_rate=0.01,
                              metric='multi_logloss')

W_4 = np.concatenate((model_4.predict([data_train, data_train]), df.agency.values[train].reshape(-1, 1)), axis=1)
Z_4 = np.concatenate((model_4.predict([data_test, data_test]), df.agency.values[test].reshape(-1, 1)), axis=1)
lgbclf_4.fit(W_4, y[train])
# joblib.dump(lgbclf_4, '../models/lgbclf_4')
raw_lgbm_predictions_4 = lgbclf_4.predict_proba(Z_4)
class_lgbm_predictions_4 = [np.argmax(x) for x in raw_lgbm_predictions_4]
print("Results for LGBM Classifier on top of Model #4")
print(classification_report(y[test], le.inverse_transform(class_lgbm_predictions_4)))
fig, ax = plt.subplots()
plot_tsne(ax, lgbclf_4, Z_4, y[test], title='LightGBM on CNN with static & dynamic word embeddings', raw_preds=raw_lgbm_predictions_4)
plt.savefig('../images/tSNE/2D/lgbm_cnn_4_tsne_2d.png')