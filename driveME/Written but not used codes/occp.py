# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:29:50 2019

@author: Kazım
"""
import h5py
import numpy as np
 
f = h5py.File('occp.h5','r')
a=f['X'][:,:]
b=f['Y'][:,:]
f.close()
f = h5py.File('occp2.h5','r')
c=f['X'][:,:]
d=f['Y'][:,:]
f.close()
f = h5py.File('occp3.h5','r')
e=f['X'][:,:]
f=f['Y'][:,:]
f.close()

x1 = a[:55980,:]
y1 = b[:55980,:]
x2 = c[:79610,:]
y2 = d[:79610,:]
x3 = e[:79387,:]
y3 = f[:79387,:]

del a,b,c,d,e,f

x = np.concatenate([x1,x2,x3])
y = np.concatenate([y1,y2,y3])

del x1,x2,x3,y1,y2,y3

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.15,random_state=3)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K

model = Sequential()
model.add(Dense(12,input_dim=12, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12, activation='relu'))
model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
batch_size = 100
nb_epoch = 50
lr = 0.01
def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

from keras.callbacks import EarlyStopping, ModelCheckpoint
model.fit(X_train, dummy_y,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_split=0.2,
          verbose =1,
          shuffle=True,
          callbacks=[EarlyStopping(monitor='val_loss', patience=6),LearningRateScheduler(lr_schedule),
                    ModelCheckpoint('occp_model.h5',save_best_only=True)]
            )


Y_pred = model.predict(X_test)
Y_pred = Y_pred.argmax(axis=1)

#make the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)