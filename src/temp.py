import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

df = pd.read_csv('../data/model_features.csv')
df['target'] = pd.read_csv('../data/target.csv')['0']
labels = {'done': 'g',
          'partial': 'g',
          'no_docs': 'k',
          'fix': 'r',
          'rejected': 'r'}
colors = df['target'].apply(lambda x: labels[x])

svd = TruncatedSVD(n_components=20)
tsne = TSNE(n_components=2)
idxs = np.random.choice(len(df), size=1000, replace=False)
scatter = tsne.fit_transform(
    svd.fit_transform(df.drop(['Unnamed: 0', 'target'], axis=1).values[idxs])
)
xs = scatter[:, 0]
ys = scatter[:, 1]

fig, ax = plt.subplots()

ax.scatter(xs[idxs], ys[idxs], c=colors[idxs])
ax.set_title("TF/IDF similarities of FOIA request bodies visualized with t-SNE")

plt.savefig('../images/tSNE/2D/tfidf_tsne_2d.png')