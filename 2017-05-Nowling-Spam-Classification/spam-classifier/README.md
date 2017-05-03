# spam-classifier
Example scripts for performing spam classification using the [trec07p](http://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html) dataset.

## Instructions
You'll need to download the [trec07p](http://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html) dataset and extract it.  There are then two scripts for spam classification: the basic spam classifier used to generate the figures in the first part of the talk and a modified version that employs feature hashing.  To run them:

```
$ python spam_classifier.py \
    --trec-dir trec07p \
    --figures-dir figures
```

A `figures` directory will be created for storing the plots.  A `cache` directory will also be created to make running the script faster, in case you want to try changing the classifier.

To run the version with feature hashing:

```
$ python spam_classifier_feature_hashing.py \
   --trec-dir trec07p \
    --figures-dir figures
```

The third Python file (`email_parsing.py`) contains functions for parsing the emails in the trec07p dataset that are used by both scripts.

## Additional Resources
If you want to learn more, I would suggest looking at:

* [Vowpal Wabbit](http://hunch.net/~vw/), which implements feature hashing and online learning
* Scikit-learn's [FeatureHasher](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) class
* Google's [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)
