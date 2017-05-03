"""
Script for comparing spam classification with a bag-of-words model constructed with and without hashing. You'll need to download a copy of the dataset from http://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html .

Copyright 2016 Ronald J. Nowling

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from collections import defaultdict
import time

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from email_parsing import stream_email

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trec-dir",
                        required=True,
                        type=str,
                        help="Location of trec07p dataset")

    parser.add_argument("--figures-dir",
                        required=True,
                        type=str,
                        help="Directory for saving figures")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)
    
    training_size = int(75419. * 0.75) # from Attenberg paper
    stream = stream_email(args.trec_dir)

    # read and parse training set
    training_bodies = []
    training_labels = []
    validation_bodies = []
    validation_labels = []
    begin_time = time.clock()
    print "Reading training set"
    for idx, (label, to, from_, body) in enumerate(stream):
        if idx < training_size:
            training_bodies.append(body)
            training_labels.append(label)
        else:
            validation_bodies.append(body)
            validation_labels.append(label)
    end_time = time.clock()
    elapsed_sec = end_time - begin_time
    print "Dataset took %s s to read" % elapsed_sec
    print

    print "Vectorizing with Dictionary"
    begin_time = time.clock()
    tfidf_vectorizer = TfidfVectorizer(binary=True, norm=None, use_idf=False)
    training_dict_features = tfidf_vectorizer.fit_transform(training_bodies)
    end_time = time.clock()
    elapsed_sec = end_time - begin_time
    print "Vectorizing took %s s" % elapsed_sec
    print
    
    print "Training on dictionary-encoded features"
    begin_time = time.clock()
    tfidf_lr = SGDClassifier(loss="log", penalty="l2", n_iter=20)
    tfidf_lr.fit(training_dict_feat, training_labels)
    end_time = time.clock()
    elapsed_sec = end_time - begin_time
    print "Training took %s s" % elapsed_sec
    print


    print "Evaluating predictions on dictionary-encoded features"
    begin_time = time.clock()
    validation_dict_features = tfidf_vectorizer.transform(validation_bodies)
    end_time = time.clock()
    elapsed_sec = end_time - begin_time
    print "Vectorizing (dict) validation set took %s s" % elapsed_sec
    print

    begin_time = time.clock()
    dict_pred_probs = tfidf_lr.predict_proba(validation_features)
    end_time = time.clock()
    elapsed_sec = end_time - begin_time
    print "Predictions (dict) on validation set took %s s" % elapsed_sec
    print

    dict_auc = roc_auc_score(validation_labels, dict_pred_probs[:, 1])

    n_dict_features = validation_dict_features.shape[1]
    print "dict auc", dict_auc, "n_features", n_dict_features
    print

    aucs = []
    nzs = []
    bit_range = list(range(8, 25))
    for n_bits in bit_range:
        lr = SGDClassifier(loss="log", penalty="l2", n_iter=20)
        hashing_vectorizer = HashingVectorizer(n_features = 2 ** n_bits, binary=True, norm=None)

        print "Vectorizing training set with 2**%s hashed features" % n_bits
        begin_time = time.clock()
        hashed_training_features = hashing_vectorizer.transform(training_bodies)
        end_time = time.clock()
        elapsed_sec = end_time - begin_time
        print "Vectorizing took %s sec" % elapsed_sec
        print

        print "Training with 2**%s hashed features" % n_bits
        begin_time = time.clock()
        lr.fit(hashed_training_features, training_labels)
        end_time = time.clock()
        elapsed_sec = end_time - begin_time
        print "Training took %s sec" % elapsed_sec
        print

        print "Vectorizing validation set with 2**%s hashed features" % n_bits
        begin_time = time.clock()
        hashed_validation_features = hashing_vectorizer.transform(validation_bodies)
        end_time = time.clock()
        elapsed_sec = end_time - begin_time
        print "Vectorizing took %s sec" % elapsed_sec
        print

        print "Prediction in validation set with 2**%s hashed features" % n_bits
        begin_time = time.clock()
        pred_probs = lr.predict_proba(hashed_validation_features)
        end_time = time.clock()
        elapsed_sec = end_time - begin_time
        print "Prediction took %s sec" % elapsed_sec
        print

        aucs.append(roc_auc_score(validation_labels, pred_probs[:, 1]))
        nzs.append((lr.coef_ != 0).sum())

        print n_bits, aucs[-1]
    
    fig, ax1 = plt.subplots()
    ax1.plot(bit_range, aucs, 'c-')
    ax1.plot(bit_range, [tfidf_auc] * len(bit_range), 'c--', label="Tfidf")
    ax1.set_xlabel('Hashed Features (log_2)', fontsize=16)
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('AUC', color='c', fontsize=16)
    for tl in ax1.get_yticklabels():
        tl.set_color('c')

    ax2 = ax1.twinx()
    ax2.plot(bit_range, nzs, 'k-')
    ax2.plot(bit_range, [n_tfidf_features] * len(bit_range), 'k--')
    ax2.set_ylabel('Non-zero Weights', color='k', fontsize=16)
    for tl in ax2.get_yticklabels():
        tl.set_color('k')

    fig.subplots_adjust(right=0.8)
    fig.savefig(os.path.join(args.figures_dir,
                             "hashed_features_auc_weights.png"),
                DPI=200)
