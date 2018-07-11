import numpy as np
import itertools
import operator
import nltk
import csv
import sys
from datetime import datetime
from .utils import *
import matplotlib.pyplot as plt


unk = 'UNK'
start = '<s>'
end = '<e>'

print("Reading CSV file")
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # split data into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # appending start and end tags
    sentences = ["{} {} {}".format(start, s, end) for s in sentences]

print("Parsed {} sentences".format(len(sentences)))

# getting tokens for each sentence
token_sentences = [nltk.word_tokenize(s) for s in sentences]

# Getting word frequencies(unigram index)
word_freqs = nltk.FreqDist(itertools.chain(*token_sentences))
print("Found {} unique words".format(len(word_freqs.keys())))

# Replacing words with freq < 5 as unk
min_freq = 5
unknown_words = set()
for i, sentences in enumerate(token_sentences):
    new_sent = []
    for word in token_sentences[i]:
        if word_freqs[word] >= min_freq:
            new_sent.append(word)
        else:
            new_sent.append(unk)
            word_freqs[unk] = word_freqs.get(word) + word_freqs.get(unk, 0)
            unknown_words.add(word)
    token_sentences[i] = new_sent

vocabulary = list(set(word_freqs.keys()).difference(unknown_words))
word_to_index = dict([(w, i) for i, w in enumerate(vocabulary)])

print("Vocabulary size {}".format(len(vocabulary)))
print("Example sentence {}".format(token_sentences[0]))

# Creating training data and labels
X_train = np.asarray([[[[word_to_index[w] for w in sent[:-1]] for sent in token_sentences]]])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in token_sentences])



