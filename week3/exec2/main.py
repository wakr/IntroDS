import pandas as pd
import scipy as sp
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
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
logreg = linear_model.LogisticRegression(C=1e5, multi_class='ovr')

logreg.fit(tr_X, tr_Y)

heur_Y = np.bincount(tr_Y.flatten().tolist()[0]).argmax()

Z = logreg.predict(ts_X)

accuracy_score(ts_Y, Z)
accuracy_score(ts_Y, [heur_Y]*ts_Y.size)


#%%
misclas = [i for (i, z) in enumerate(Z) if ts_Y[i] != Z[i]]
mc_i = misclas[0]
sp.misc.imshow(np.reshape(ts_X[mc_i], (32,32)))



