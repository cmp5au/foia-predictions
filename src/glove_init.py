import numpy as np
from scipy import stats
from keras.layers import Embedding

# The imports below are just for the if name==main block
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

def create_embeddings_matrix(vocabulary, embedding_dim=100):
    '''
    Create a matrix of word embeddings initialized with the GloVe embeddings,
    then initialize new words by sampling from a uniform distribution with
    the same variance as the pre-trained words in each dimension.
    Default is using 100-dimensional vectors, e.g. glove.6B.100d.txt
    '''
    embeddings_index = {}
    with open('path/to/your/glove_vectors.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((len(vocabulary) + 1, embedding_dim))
    untrained_indices = np.ones((len(vocabulary) + 1,), bool)
    for i, word in enumerate(vocabulary):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            untrained_indices[i] = False
            embeddings_matrix[i] = embedding_vector

    stds = np.std(embeddings_matrix[~untrained_indices, :], axis=0)

    for i in range(len(vocabulary)):
        if untrained_indices[i]:
            embeddings_matrix[i] = np.array([stats.uniform(-s, s).rvs() * np.sqrt(3)
                                             for s in stds])
    print('Matrix shape: {}'.format(embeddings_matrix.shape))
    return embeddings_matrix

def create_embeddings_layer(embeddings_matrix, name, max_len, trainable=False):
    '''
    Create a layer of word embeddings for a keras-based NN
    '''
    embedding_layer = Embedding(
        input_dim=embeddings_matrix.shape[0],
        output_dim=embeddings_matrix.shape[1],
        input_length=max_len,
        weights=[embeddings_matrix],
        trainable=trainable,
        name=name)
    return embedding_layer

if __name__ == '__main__':
    df = pd.read_csv('../data/body_multiclass_target.csv')
    df.dropna(axis=0, inplace=True)
    corpus = df.body.values
    max_len = 250 # caps the maximum token number at 250 (250 words per document)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    embeddings_matrix = create_embeddings_matrix(tokenizer.word_index)

    # use a static layer with trainable=False for basic embedding
    embedding_layer_static = create_embeddings_layer(embeddings_matrix,
                                'embedding_layer_static',
                                max_len,
                                trainable=False)

    # use a dynamic layer with trainable=True if you want to train your
    # embeddings on your corpus
    embedding_layer_dynamic = create_embeddings_layer(embeddings_matrix,
                                'embedding_layer_dynamic',
                                max_len,
                                trainable=True)