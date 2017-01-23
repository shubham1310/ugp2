import os
import matplotlib.pyplot as plt
import math, pickle
import numpy as np
from sklearn import metrics
import itertools
from random import shuffle
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from collections import Counter


path = './Transcripts/'
ps = pickle.load(open(path+'scenes.txt', "rb"))
charac=['joey','ross','chandler','monica','rachel','phoebe','other','end']
data=pickle.load(open('../../skip-thoughts/skipvectorscenes.txt','r'))
X=[]
Y=[]

def onehotcharac(value):
	temp=np.zeros(len(charac))
	temp[charac.index(value)]=1
	return temp

for x in data:
	temX=[]
	temY=[]
	for j in range(len(x)) :
		temX.append(np.concatenate((onehotcharac(x[j][0]),x[j][1]),axis=0))
		if j==len(x):
			temY.append(onehotcharac('end'))
		else:
			temY.append(onehotcharac(x[j][0]))
		# temY.append()
	X.append(temX)
	Y.append(temY)

Xtrain = np.array(X[:int(0.7*len(X))])
ytrain = np.array(Y[:int(0.7*len(Y))])
Xtest = np.array(X[int(0.7*len(X)):])
ytest = np.array(Y[int(0.7*len(Y)):])

print(len(Xtrain), 'train sequences')
print(len(Xtest), 'test sequences')


maxlen=max([len(i) for i in data])
batchsize =32
max_features = 20000
Xlen = len(X[0][0])

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(Xtrain, maxlen=maxlen)
X_test = sequence.pad_sequences(Xtest, maxlen=maxlen)
y_train = sequence.pad_sequences(ytrain, maxlen=maxlen)
y_test = sequence.pad_sequences(ytest, maxlen=maxlen)
print Xtrain
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)



print('Build model...')
model = Sequential()
# model.add(Embedding(max_features, 512, dropout=0.2))
model.add(LSTM(output_dim=len(charac),return_sequences =True,input_shape=(maxlen, Xlen)))
# model.add(Dense(output_dim=len(charac)))
# model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# for iteration in range(1, 60):
	# print'Iteration', iteration
model.fit(X_train, y_train, batch_size=batchsize, nb_epoch=100, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test)
print('Test score:', score)
	# print('Test accuracy:', acc)
