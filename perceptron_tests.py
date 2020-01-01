from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from data_preprocessing import *

print("Perceptron Classifier, Average 3-Folds Cross Validation Tests")

# ---------Unigrams Frequence Perceptron
clf_unif_Perceptron = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=False)),
                                ("tfid", TfidfTransformer()), ("clf", Perceptron())])
scores_unif_Perceptron = cross_val_score(clf_unif_Perceptron, data, target, cv=3)
print("Unigrams (frequency) - Accuracy: %0.2f (+/- %0.2f)" % (scores_unif_Perceptron.mean(),
                                                            scores_unif_Perceptron.std() * 2))

# ----------Unigrams Presence Perceptron
clf_unip_Perceptron = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", Perceptron())])
scores_unip_Perceptron = cross_val_score(clf_unip_Perceptron, data, target, cv=3)
print("Unigrams (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_unip_Perceptron.mean(),
                                                            scores_unip_Perceptron.std() * 2))

# ---------Unigrams + Bigrams Presence Perceptron
clf_unibigp_Perceptron = Pipeline([("vect", CountVectorizer(ngram_range=(1, 2), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", Perceptron())])
scores_unibigp_Perceptron = cross_val_score(clf_unibigp_Perceptron, data, target, cv=3)
print("Unigrams + bigrams (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_unibigp_Perceptron.mean(),
                                                            scores_unibigp_Perceptron.std() * 2))

# ---------Bigrams Presence Perceptron
clf_bigp_Perceptron = Pipeline([("vect", CountVectorizer(ngram_range=(2, 2), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", Perceptron())])
scores_bigp_Perceptron = cross_val_score(clf_bigp_Perceptron, data, target, cv=3)
print("Bigrams (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_bigp_Perceptron.mean(),
                                                            scores_bigp_Perceptron.std() * 2))

# ----------Top 2633 Unigrams Presence Perceptron
clf_topunip_Perceptron = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True, max_features=2633)),
                                ("tfid", TfidfTransformer()), ("clf", Perceptron())])
scores_topunip_Perceptron = cross_val_score(clf_topunip_Perceptron, data, target, cv=3)
print("Top 2633 unigrams (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_topunip_Perceptron.mean(),
                                                            scores_topunip_Perceptron.std() * 2))

# ----------Unigrams + POS Presence Perceptron
clf_posunip_Perceptron = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", Perceptron())])
scores_posunip_Perceptron = cross_val_score(clf_posunip_Perceptron, pos_data, target, cv=3)
print("Unigrams + POS tags (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_posunip_Perceptron.mean(),
                                                            scores_posunip_Perceptron.std() * 2))

# ----------Adjectives Unigrams Presence Perceptron
clf_adjunip_Perceptron = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", Perceptron())])
scores_adjunip_Perceptron = cross_val_score(clf_adjunip_Perceptron, adj_data, target, cv=3)
print("Adjectives (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_adjunip_Perceptron.mean(),
                                                            scores_adjunip_Perceptron.std() * 2))

# ----------Unigrams + position Presence Perceptron
clf_positunip_Perceptron = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", Perceptron())])
scores_positunip_Perceptron = cross_val_score(clf_positunip_Perceptron, position_data, target, cv=3)
print("Unigrams + position (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_positunip_Perceptron.mean(),
                                                            scores_positunip_Perceptron.std() * 2))
