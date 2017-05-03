"""
Script for comparing Logistic Regression with L1, L2, and elastic net regularization and the liblinear, sag, and sgd optimizers. You'll need to download a copy of the dataset from http://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html .

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
import cPickle
from itertools import islice
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score

from email_parsing import read_all_emails

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

    parser.add_argument("--cache-dir",
                        type=str,
                        default="cache",
                        help="Speed up repeated runs of the script")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    if not os.path.exists(args.figures_dir):
        os.makedirs(args.figures_dir)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    training_size = int(75419. * 0.75) # from Attenberg paper
    # try loading features from the cache
    if os.path.exists(os.path.join(args.cache_dir, "_success")):
        training_labels = np.load(os.path.join(args.cache_dir,
                                               "training_labels.npy"))
        validation_labels = np.load(os.path.join(args.cache_dir,
                                                 "validation_labels.npy"))
        training_features = spio.mmread(os.path.join(args.cache_dir,
                                                     "training_features.mtx"))
        validation_features = spio.mmread(os.path.join(args.cache_dir,
                                                       "validation_features.mtx"))
        with open(os.path.join(args.cache_dir, "vectorizer.pkl")) as fl:
            vectorizer = cPickle.load(fl)

    else:
        vectorizer = TfidfVectorizer(binary=True, norm=None, use_idf=False)
        print "Reading and parsing emails"
        begin_time = time.clock()
        bodies, labels = read_all_emails(args.trec_dir)
        end_time = time.clock()
        elapsed_sec = end_time - begin_time
        print "Dataset took %s s to read and parse" % elapsed_sec
        print

        print "Vectorizing Training Set"
        begin_time = time.clock()
        training_features = vectorizer.fit_transform(bodies[:training_size])
        end_time = time.clock()
        elapsed_sec = end_time - begin_time
        print "Took %s s to build dictionary and encode features" % elapsed_sec
        print

        print "Vectorizing Validation Set"
        begin_time = time.clock()
        validation_features = vectorizer.transform(bodies[training_size:])
        end_time = time.clock()
        elapsed_sec = end_time - begin_time
        print "Took %s s to vectorize validation set" % elapsed_sec
        print

        training_labels = np.array(labels[:training_size])
        validation_labels = np.array(labels[training_size:])
        
        np.save(os.path.join(args.cache_dir,
                             "training_labels.npy"), training_labels)
        np.save(os.path.join(args.cache_dir,
                             "validation_labels.npy"), validation_labels)
        spio.mmwrite(os.path.join(args.cache_dir,
                                  "training_features.mtx"), training_features)
        spio.mmwrite(os.path.join(args.cache_dir,
                                  "validation_features.mtx"), validation_features)
        with open(os.path.join(args.cache_dir, "vectorizer.pkl"), "w") as fl:
            cPickle.dump(vectorizer, fl)

        with open(os.path.join(args.cache_dir, "_success"), "w") as fl:
            fl.write("cached\n")

    print "Training"
    sgd_l2 = SGDClassifier(loss="log", penalty="l2", n_iter=20)
    begin_time = time.clock()
    sgd_l2.fit(training_features, training_labels)
    end_time = time.clock()
    elapsed_sec = end_time - begin_time
    print "Took %s s to train classifiers" % elapsed_sec
    print
    
    print "Predicting"
    begin_time = time.clock()
    pred_probs_sgd_l2 = sgd_l2.predict_proba(validation_features)
    end_time = time.clock()
    elapsed_sec = end_time - begin_time
    print "Took %s s to predict probabilities" % elapsed_sec
    print

    plt.clf()
    fpr, tpr, _ = roc_curve(validation_labels, pred_probs_sgd_l2[:, 1], pos_label=1)
    plt.plot(fpr, tpr, label="SGD L2")
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.xlim([0.0, 0.1])
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.figures_dir, "roc_curve.png"), DPI=300)

    n_features = validation_features.shape[1]
    print "features", n_features
    print "AUC", roc_auc_score(validation_labels, pred_probs_sgd_l2[:, 1])

    plt.clf()
    pos_probs = []
    neg_probs = []
    for i, label in enumerate(validation_labels):
        prob = pred_probs_sgd_l2[i, 1]
        if label == 0.:
            neg_probs.append(prob)
        else:
            pos_probs.append(prob)
    bins = np.linspace(start=0., stop=1., num=21.)
    plt.hist(pos_probs,
             bins=bins,
             color="k",
             alpha=0.7,
             label="Spam")
    plt.hist(neg_probs,
             bins=bins,
             color="c",
             alpha=0.7,
             label="Ham")
    plt.xlabel("Probability", fontsize=16)
    plt.ylabel("Occurrences (Emails)", fontsize=16)
    plt.xlim([0.0, 1.0])
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(args.figures_dir, "lr_prob_hist.png"), DPI=300)

    sorted_weights = list(sorted(sgd_l2.coef_[0, :], reverse=True))
    plt.clf()
    plt.plot(sorted_weights, color="m")
    plt.ylabel("Weight", fontsize=16)
    plt.xlabel("Feature", fontsize=16)
    plt.savefig(os.path.join(args.figures_dir, "lr_feature_weights.png"), DPI=300)

    sorted_indices = np.argsort(sgd_l2.coef_[0, :])
    n_features = sgd_l2.coef_.shape[1]
    reverse_map = { idx : word for word, idx in vectorizer.vocabulary_.iteritems() }
    for i in xrange(1, 21):
        idx = sorted_indices[n_features - i]
        print i, sgd_l2.coef_[0, idx], reverse_map[idx]
    
            
