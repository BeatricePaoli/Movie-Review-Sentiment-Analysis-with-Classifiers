# Movie Review Sentiment Analysis with Classifiers

This is a project for a AI Course of a Bachelor degree in Computer Engineering of the University of Florence, Italy. 
It is about testing two Classifiers for Machine Learning, Perceptron and Decision Tree, using the implementation from SciKit-Learn.
The goal is to make sentiment analysis of movie reviews. The dataset used is the polarity_dataset v2.0 from Bo Pang and Lillian Lee ([link](http://www.cs.cornell.edu/people/pabo/movie-review-data/)).

## Setup

The project was created for Python 3. To run the code the environment needs the following additional packages:

* SciKit-Learn
* Numpy
* ScyPy
* NLTK

The dataset also needs to be download in the root directory of the project with the structure of the subfolders

```
txt_sentoken/neg # for the negative reviews
txt_sentoken/pos # for the positive reviews
```

## Running the tests

For the tests on the Perceptron classifier run perceptron_tests.py 
For the tests on the Decision Tree classifier run decision_tree_tests.py 

The results are the average accuracies on 3-Folds Cross Validation tests done on different kinds of features (unigrams, bigrams, unigrams with POS tagging etc.).

## Authors

* **Beatrice Paoli**

## References

* [Thumbs up? Sentiment Classification using Machine Learning Techniques](http://www.cs.cornell.edu/home/llee/papers/sentiment.pdf) - Bo Pang, Lillian Lee (Proceedings of EMNLP, 2002)