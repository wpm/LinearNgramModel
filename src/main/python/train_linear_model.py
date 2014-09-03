"""
Train a linear n-gram model with tf-idf features and serialize it.

The training file contains an integer label followed by tab followed by a text document.

The gzipped model file this script creates can be used as input to ApplyModel.java.
"""

import gzip

import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


def train_model(corpus):
    labels = []
    data = []
    for line in corpus:
        label, vector = line.decode('utf-8').split("\t")
        labels.append(int(label))
        data.append(vector)
    model = Pipeline([('vect', CountVectorizer(ngram_range=[1, 2])),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])
    model.fit(data, labels)
    return model


def serialize(model, filename):
    vect = model.named_steps['vect']
    vocabulary = vect.vocabulary_
    ngrams = vect.ngram_range
    tfidf = model.named_steps['tfidf']
    idf = tfidf.idf_.tolist()
    clf = model.named_steps['clf']
    weights = [flp.tolist() for flp in clf.feature_log_prob_]
    biases = clf.class_log_prior_.tolist()
    with gzip.open(filename, 'w') as f:
        json.dump({"vocabulary": vocabulary,
                   "ngrams": ngrams,
                   "idf": idf,
                   "weights": weights,
                   "biases": biases}, f,
                  indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('corpus', type=argparse.FileType(), help="labeled training data")
    parser.add_argument('model', help="gzipped model")
    args = parser.parse_args()
    model = train_model(args.corpus)
    serialize(model, args.model)
