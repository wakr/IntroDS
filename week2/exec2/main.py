import pandas as pd
from collections import Counter
import nltk
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# 1

pos = pd.read_table("week2/exec2/pos.txt", header=None, squeeze=True)
neg = pd.read_table("week2/exec2/neg.txt", header=None, squeeze=True)

neg_split = neg.str.replace(r'\|', ' ').str.cat(sep=' ')
pos_split = pos.str.replace(r'\|', ' ').str.cat(sep=' ')

neg_words = nltk.tokenize.word_tokenize(neg_split)
neg_word_dist = nltk.FreqDist(neg_words)

pos_words = nltk.tokenize.word_tokenize(pos_split)
pos_word_dist = nltk.FreqDist(pos_words)

print(neg_word_dist.most_common(10))
print(pos_word_dist.most_common(10))


# 2-4
from sklearn.feature_extraction.text import TfidfVectorizer
tf_neg = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
tf_pos = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')

tfidf_matrix_neg =  tf_neg.fit_transform([neg_split])
tfidf_matrix_pos =  tf_pos.fit_transform([pos_split])

feature_names_neg = tf_neg.get_feature_names() 
feature_names_pos = tf_pos.get_feature_names()

dense_neg = tfidf_matrix_neg.todense()[0].tolist()[0]
dense_pos = tfidf_matrix_pos.todense()[0].tolist()[0]

phrase_scores_neg = [pair for pair in zip(range(0, len(dense_neg)), dense_neg)]
phrase_scores_pos = [pair for pair in zip(range(0, len(dense_pos)), dense_pos)]

sorted_phrase_scores_neg = sorted(phrase_scores_neg, key=lambda t: t[1] * -1)
sorted_phrase_scores_pos = sorted(phrase_scores_pos, key=lambda t: t[1] * -1)


scores_as_names_neg = [(feature_names_neg[wID], s) for (wID, s) in sorted_phrase_scores_neg]
scores_as_names_pos = [(feature_names_pos[wID], s) for (wID, s) in sorted_phrase_scores_pos]

print("Top 10 neg", scores_as_names_neg[:10])
print("Top 10 pos", scores_as_names_pos[:10])

print("Top 10 neg", neg_word_dist.most_common(10))
print("Top 10 pos",pos_word_dist.most_common(10))

def make_plot(d):
    words = list(zip(*d))[0][:50]
    score = list(zip(*d))[1][:50]
    x_pos = np.arange(len(words))
    plt.figure(figsize=(30,20))
    plt.bar(x_pos, score,align='center')
    plt.xticks(x_pos, words, rotation=90)
    plt.ylabel('Popularity Score')
    plt.show()
    
make_plot(scores_as_names_neg)
make_plot(scores_as_names_pos)

