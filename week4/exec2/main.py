# Python2 compatibility
from __future__ import print_function

import numpy as np
import pandas as pd
#%%

from scipy.misc import imread

def read_data(folder):
    file = "hasy-data-labels.csv"
    df = pd.read_csv(folder + file)
    df["X"] = df.path.apply(lambda x: imread(folder+x, flatten=True))
    df = df.sample(frac=1).reset_index(drop=True) # shuffle
    y = df.symbol_id.as_matrix()
    X = []
    for i in range(df.X.size):
        img = df.X.iloc[i] 
        X.append(img)
    return np.array(X), y

X, y = read_data("week3/exec2/HASYv2/")

print(X.shape, y.shape) # Should be (168233, 32, 32) (168233,)

#%%

def split_data(X, y):
    
    total_elems = y.size
    train_size = int(0.8 * total_elems)
    X_train = X[:train_size, :, :]
    y_train = y[:train_size]
    
    left_over_X = X[train_size: ,:, :]
    left_over_y = y[train_size:]
    total_left_elems = left_over_y.size
    
    val_size = int(0.5 * total_left_elems) # (0.8, 0.2) --> (0.1, 0.1)
    X_val = left_over_X[:val_size, :, :]
    y_val = left_over_y[:val_size]
    
    X_test = left_over_X[val_size:, :, :]
    y_test = left_over_y[val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
    
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

print(X_train.shape, y_train.shape) # Should yield approx (134586, 32, 32) (134586,)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
#%%

# Give new ids to classes such that the first unique symbol_id gets the number 0,
# the next unique symbol_id gets the number 1, and so forth
def transform_labels(y):
    uniq = np.unique(y)
    look_up = {}
    for (i, x) in enumerate(uniq):
        look_up[x] = i 
    
    res = [look_up[x] for x in y]
    
    return np.array(res)

y_train, y_val, y_test = map(transform_labels, [y_train, y_val, y_test])

print(y_train.shape, y_val.shape, y_test.shape) # Should be approx (134586,) (16823,) (16824,)

# Should return the elements in arr for which their corresponding label in y_arr is in between [0, 100]
def filter_out(arr, y_arr):
    elems_count = y_arr.size
    res_X = []
    res_Y = []
    for i in range(elems_count):
        if 0 <= y_arr[i] <= 100:
            res_Y.append(y_arr[i])
            res_X.append(arr[i,:,:])
    return np.array(res_X), np.array(res_Y)
        
        

X_train, y_train = filter_out(X_train, y_train)
X_val, y_val = filter_out(X_val, y_val)
X_test, y_test = filter_out(X_test, y_test)

print(y_train.shape, X_train.shape) # Should be approx (34062,) (34062, 32, 32)


#%%

# Convert labels to one-hot encoding here 
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print(y_train.shape) # Should be approx (34062, 100)


#%%

from keras.models import Sequential
from keras.layers import Dense, Flatten

# This function should return a keras Sequential model that has the appropriate layers
def create_linear_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32)))
    model.add(Dense(101, activation="softmax"))
    return model

model = create_linear_model()
model.summary()

#%%

# Feel free to try out other optimizers. Categorical crossentropy loss means 
# we are predicting the probability of each class separately.
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=3, batch_size=64)

#%% 

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.backend import clear_session

def create_convolutional_model():
    model = Sequential()
    
    model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 1))) # A convolutional layer
    model.add(MaxPooling2D((4,4))) # Max pooling reduces the complexity of the model
    model.add(Dropout(0.4)) # Randomly dropping connections within the network helps against overfitting
    model.add(Conv2D(128, (2, 2), activation="relu")) 
    model.add(BatchNormalization()) # Numbers within the network might get really big, so normalize them 
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(y_train.shape[1], activation="softmax"))
    
    return model

clear_session()

model = create_convolutional_model()
model.summary() # Get a summary of all the layers

#%%

# Feel free to try out other optimizers. Categorical crossentropy loss means 
# we are predicting the probability of each class separately.
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Extra axis for "gray" channel
model.fit(X_train[:, :, :, np.newaxis], y_train, epochs=5, batch_size=64, validation_data=(X_val[:, :, :, np.newaxis], y_val))


#%%

model.evaluate(X_test[:, :, :, np.newaxis], y_test)

