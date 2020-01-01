from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from data_preprocessing import *

print("Decision Tree Classifier, Average 3-Folds Cross Validation Tests")

# ---------Unigrams Frequence Decision Tree
clf_unif_DT = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=False)),
                                ("tfid", TfidfTransformer()), ("clf", DecisionTreeClassifier())])
scores_unif_DT = cross_val_score(clf_unif_DT, data, target, cv=3)
print("Unigrams (frequency) - Accuracy: %0.2f (+/- %0.2f)" % (scores_unif_DT.mean(), scores_unif_DT.std() * 2))

# ----------Unigrams Presence Decision Tree
clf_unip_DT = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", DecisionTreeClassifier())])
scores_unip_DT = cross_val_score(clf_unip_DT, data, target, cv=3)
print("Unigrams (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_unip_DT.mean(), scores_unip_DT.std() * 2))

# ---------Unigrams + Bigrams Presence Decision Tree
clf_unibigp_DT = Pipeline([("vect", CountVectorizer(ngram_range=(1, 2), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", DecisionTreeClassifier())])
scores_unibigp_DT = cross_val_score(clf_unibigp_DT, data, target, cv=3)
print("Unigrams + bigrams (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_unibigp_DT.mean(),
                                                                       scores_unibigp_DT.std() * 2))

# ---------Bigrams Presence Decision Tree
clf_bigp_DT = Pipeline([("vect", CountVectorizer(ngram_range=(2, 2), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", DecisionTreeClassifier())])
scores_bigp_DT = cross_val_score(clf_bigp_DT, data, target, cv=3)
print("Bigrams (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_bigp_DT.mean(), scores_bigp_DT.std() * 2))

# ----------Top 2633 Unigrams Presence Decision Tree
clf_topunip_DT = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True, max_features=2633)),
                                ("tfid", TfidfTransformer()), ("clf", DecisionTreeClassifier())])
scores_topunip_DT = cross_val_score(clf_topunip_DT, data, target, cv=3)
print("Top 2633 unigrams (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_topunip_DT.mean(),
                                                            scores_topunip_DT.std() * 2))

# ----------Unigrams + POS Presence Decision Tree
clf_posunip_DT = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", DecisionTreeClassifier())])
scores_posunip_DT = cross_val_score(clf_posunip_DT, pos_data, target, cv=3)
print("Unigrams + POS tags (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_posunip_DT.mean(),
                                                                        scores_posunip_DT.std() * 2))

# ----------Adjectives Unigrams Presence Decision Tree
clf_adjunip_DT = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", DecisionTreeClassifier())])
scores_adjunip_DT = cross_val_score(clf_adjunip_DT, adj_data, target, cv=3)
print("Adjectives (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_adjunip_DT.mean(), scores_adjunip_DT.std() * 2))

# ----------Unigrams + position Presence Decision Tree
clf_positunip_DT = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), binary=True)),
                                ("tfid", TfidfTransformer()), ("clf", DecisionTreeClassifier())])
scores_positunip_DT = cross_val_score(clf_positunip_DT, position_data, target, cv=3)
print("Unigrams + position (presence) - Accuracy: %0.2f (+/- %0.2f)" % (scores_positunip_DT.mean(),
                                                                        scores_positunip_DT.std() * 2))
