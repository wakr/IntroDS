# -*- coding: utf-8 -*-

import pandas as pd
import scipy as sp
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('exec2/HASYv2/hasy-data-labels.csv')
df = df.query("70 <= symbol_id <= 80")
df
#%%

y = df.symbol_id.tolist()

paths = df.path.tolist()

X = []

for p in paths: 
    img = sp.misc.imread("exec2/HASYv2/"+p, flatten=True).flatten()
    X.append(img)
#%%

data = []

for i in range(len(y)):
    data.append((X[i], y[i]))
    
df2 = pd.DataFrame(data, columns=['X', 'Y'])
df2['X'] = df2['X'].tolist()
df2['Y'] = df2['Y'].astype('category')
df2 = df2.sample(frac=1)

train, test = train_test_split(df2, test_size=0.2)
#%%
def to_matrix(dataf):
    res = []
    for i in range(dataf.shape[0]):
        d = np.append(dataf.X.iloc[i], dataf.Y.iloc[i])
        res.append(d)
    res = np.asmatrix(res)
    return res
    
#%%    
    
data_tr = []
for i in range(train.shape[0]):
    d = np.append(train.X.iloc[i], train.Y.iloc[i])
    data_tr.append(d)
data_tr = np.asmatrix(data_tr)

tr_X = data_tr[:,:-1]
tr_Y = data_tr[:,-1]

data_ts = []
for i in range(test.shape[0]):
    d = np.append(test.X.iloc[i], test.Y.iloc[i])
    data_ts.append(d)
data_ts = np.asmatrix(data_ts)

ts_X = data_ts[:,:-1]
ts_Y = data_ts[:,-1]
#%%

rf = RandomForestClassifier()
rf.fit(tr_X, tr_Y)

heur_Y = np.bincount(tr_Y.flatten().tolist()[0]).argmax()

Z = rf.predict(ts_X)

accuracy_score(ts_Y, Z) # slightly worse result

#%%

forests = [RandomForestClassifier(n_estimators=i*10) for i in range(1,21)]
train_forests = [m.fit(tr_X, tr_Y) for m in forests]
#%%

# more trees will get better result
accuracies = [accuracy_score(ts_Y, m.predict(ts_X)) for m in train_forests]

plt.plot([i*10 for i in range(1,21)], accuracies)
plt.show()


#%%

# to good to learn anything from test data as it should be only using for 
# testing purposes. Maybe learning from train data could be more suitable


#%% reshuffle and resample
df2 = df2.sample(frac=1)

train, test = train_test_split(df2, test_size=0.1)
train, validation = train_test_split(train, test_size=0.1)

data_tr = to_matrix(train)
tr_X = data_tr[:,:-1]
tr_Y = data_tr[:,-1]

data_ts = to_matrix(test)
ts_X = data_ts[:,:-1]
ts_Y = data_ts[:,-1]

data_tv = to_matrix(validation)
tv_X = data_tv[:,:-1]
tv_Y = data_tv[:,-1]




#%% 

forests = [RandomForestClassifier(n_estimators=i*10) for i in range(1,21)]
train_forests = [m.fit(tr_X, tr_Y) for m in forests]

accuracies = [accuracy_score(tv_Y, m.predict(tv_X)) for m in train_forests]

best_model = forests[np.argmax(accuracies)]

accuracy_score(ts_Y, best_model.predict(ts_X))

# accuracy ~86% so it's pretty good. == Unseen data

#%%

from tpot import TPOTClassifier

train, test = train_test_split(df2, test_size=0.2)

data_tr = to_matrix(train)
tr_X = data_tr[:,:-1]
tr_Y = data_tr[:,-1]

data_ts = to_matrix(test)
ts_X = data_ts[:,:-1]
ts_Y = data_ts[:,-1]

#%%

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)

#%%

tpot.fit(tr_X, tr_Y)

#%%
print(tpot.score(ts_X, ts_Y))










