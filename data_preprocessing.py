import os
import numpy as np
import nltk
import re
import string
import pickle

LOAD_DATA = 0

# ------ Main Data, extracted from the dataset
def load_dataset():
    data = []
    rootdir = "txt_sentoken/neg/"
    for root, dirs, files in os.walk(rootdir):
        for filename in files:
            data.append(open(rootdir + filename).read())
    rootdir = "txt_sentoken/pos/"
    for root, dirs, files in os.walk(rootdir):
        for filename in files:
            data.append(open(rootdir + filename).read())
    return data


if LOAD_DATA:
    data = pickle.load(open("preprocessed_data/data.pickle", "rb"))
else:
    data = load_dataset()
    pickle.dump(data, open("preprocessed_data/data.pickle", "wb"))

# First half of the samples are negative reviews (labelled with 0),
# second half are positive reviews (labelled with 1)
neg_target = np.zeros(int(len(data) / 2))
pos_target = np.ones(int(len(data) / 2))
target = np.concatenate((neg_target, pos_target))


# ------ Data with POS tags
def get_POS_data(data):
    pos_data = [""]*len(data)
    for i in range(0, len(data)):
        tokenized = nltk.word_tokenize(data[i])
        tokens = nltk.pos_tag(tokenized)
        for (word, tag) in tokens:
            pos_data[i] = pos_data[i] + word + "_" + tag + " "
    return pos_data


if LOAD_DATA:
    pos_data = pickle.load(open("preprocessed_data/pos_data.pickle", "rb"))
else:
    pos_data = get_POS_data(data)
    pickle.dump(pos_data, open("preprocessed_data/pos_data.pickle", "wb"))


# ------ Data with only adjectives
def get_adj_data(data):
    adj_data = [""]*len(data)
    for i in range(0, len(data)):
        tokenized = nltk.word_tokenize(data[i])
        tokens = nltk.pos_tag(tokenized)
        for (word, tag) in tokens:
            if tag.startswith("JJ"):
                adj_data[i] = adj_data[i] + word + " "
    return adj_data


if LOAD_DATA:
    adj_data = pickle.load(open("preprocessed_data/adj_data.pickle", "rb"))
else:
    adj_data = get_adj_data(data)
    pickle.dump(adj_data, open("preprocessed_data/adj_data.pickle", "wb"))


# ------ Data with tags indicating the position in the text
def get_position_data(data):
    position_data = [""]*len(data)
    for i in range(0, len(data)):
        tot_words = len(re.findall(r"\w+", data[i]))
        no_punct_sentence = " ".join(word.strip(string.punctuation) for word in data[i].split())
        words = no_punct_sentence.split(" ")
        processed_words = []
        for word in words:
            n_occurrencies = processed_words.count(word)
            processed_words.append(word)
            ind = 0
            for k in range(0, n_occurrencies + 1):
                ind = words.index(word, ind)
                ind += 1
            word_position = ind - 1
            rel_pos = word_position / tot_words
            if rel_pos <= 0.25:
                position_data[i] = position_data[i] + word + "_" + "0" + " "
            elif rel_pos > 0.25 and rel_pos <= 0.75:
                position_data[i] = position_data[i] + word + "_" + "1" + " "
            else: # rel_pos > 0.75
                position_data[i] = position_data[i] + word + "_" + "2" + " "
    return position_data


if LOAD_DATA:
    position_data = pickle.load(open("preprocessed_data/position_data.pickle", "rb"))
else:
    position_data = get_position_data(data)
    pickle.dump(position_data, open("preprocessed_data/position_data.pickle", "wb"))
