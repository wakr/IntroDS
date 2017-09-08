import pandas as pd
import nltk

df = pd.read_json('Automotive_5.json', lines=True)

reviews = df.reviewText

# 3

reviews = reviews.str.lower()

stop_words = pd.read_table('stop-word-list.txt', header=None)[0].values.tolist()
punctuations = ['.', ',', '!', '?', '\'', '(', ')', ';', ':', '\"', '/', '-']


def replace_with_empty(s):
    res = s
    for p in punctuations:
        res = res.replace(p, "")

    splitted = [w for w in res.split() if not w in stop_words]
    res = " ".join(splitted)
    return res

reviews = reviews.apply(replace_with_empty)

sno = nltk.stem.SnowballStemmer('english')

def stemmify(s):
    res = []
    for w in s.split():
        res.append(sno.stem(w))
    return " ".join(res)


reviews = reviews.apply(stemmify)

df.reviewText = reviews

pos = df.query('overall>3').reviewText
neg = df.query('overall<3').reviewText

pos.to_csv('pos.txt', index=False, header=False)
neg.to_csv('neg.txt', index=False, header=False)
