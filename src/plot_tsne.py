import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(ax, model, X, y, dimension=2, title='', raw_preds=None):

    labels = list(set(y))
    colors = np.array(['g' if y == labels[0] else 'k' if y == labels[1] else 'r'
                        for y in y])

    if ~(raw_preds is None):
        idxs = np.random.choice(len(raw_preds), size=1000, replace=False)
    else:
        idxs = np.random.choice(range(y.shape[0]), size=1000, replace=False)
        proba_preds = model.predict_proba(X)
    

    if dimension == 2:
        tsne = TSNE(n_components=2)
        if ~(raw_preds is None):
            scatter = tsne.fit_transform(raw_preds[idxs])
        else:
            scatter = tsne.fit_transform(proba_preds[idxs])

        xs = scatter[:, 0]
        ys = scatter[:, 1]

        ax.scatter(xs, ys, c=colors[idxs])
        if title:
            ax.set_title(title)

    elif dimension == 3:
        tsne_3d = TSNE(n_components=3)
        if ~(raw_preds is None):
            scatter_3d = tsne_3d.fit_transform(raw_preds[idxs])
        else:
            scatter_3d = tsne_3d.fit_transform(proba_preds[idxs])

        xs_3d = scatter_3d[:, 0]
        ys_3d = scatter_3d[:, 1]
        zs_3d = scatter_3d[:, 2]

        ax.scatter(xs_3d, ys_3d, zs_3d, c=colors[idxs])
        if title:
            ax.set_title(title)

    else:
        raise ValueError("Dimension provided must be 2 or 3")