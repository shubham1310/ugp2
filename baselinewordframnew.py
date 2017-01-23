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
from keras.layers import Dense, Dropout, Activation, Embedding, Merge
from keras.layers import LSTM, SimpleRNN, GRU, LocallyConnected1D, TimeDistributedDense
from collections import Counter



# path = './Transcripts/'
# ps = pickle.load(open(path+'scenes.txt', "rb"))
charac=['joey','ross','chandler','monica','rachel','phoebe','other','end']
data=pickle.load(open('../../scenesskipvectorvgg.txt','r'))
X=[]
# Xfram=[]
Y=[]
print "Read the data"

def onehotcharac(value):
	temp=np.zeros(len(charac))
	temp[charac.index(value)]=1
	return temp

maxlenfram = max([ max([np.size(i[1]) for i in j]) for j in data])
for x in data:
	temX=[]
	temXfram=[]
	temY=[]
	for j in range(len(x)-1) :
		# temXfram.append()
		temX.append(np.concatenate((onehotcharac(x[j][0][0]),x[j][0][1],x[j][1]),axis=0))
		if j==len(x)-1:
			temY.append(onehotcharac('end'))
		else:
			temY.append(onehotcharac(x[j+1][0][0]))
		# temY.append()
	# Xfram.append(temXfram)
	X.append(temX)
	Y.append(temY)
print "Created X,Y"

Xtrain = np.array(X[:int(0.7*len(X))])
# Xframtrain = np.array(Xfram[:int(0.7*len(X))])
ytrain = np.array(Y[:int(0.7*len(Y))])
Xtest = np.array(X[int(0.7*len(X)):])
# Xframtest = np.array(Xfram[int(0.7*len(X)):])
ytest = np.array(Y[int(0.7*len(Y)):])

print(len(Xtrain), 'train sequences')
print(len(Xtest), 'test sequences')


maxlen=max([len(i) for i in data])
batchsize =32
max_features = 20000
Xlen = len(X[0][0])

print('Pad sequences (samples x time)')
# X_train = sequence.pad_sequences(Xtrain, maxlen=maxlen,padding='post')
# X_test = sequence.pad_sequences(Xtest, maxlen=maxlen,padding='post')
# Xframtest = sequence.pad_sequences(Xframtest, maxlen=maxlen,padding='post')
# Xframtrain = sequence.pad_sequences(Xframtrain, maxlen=maxlen,padding='post')
# y_train = sequence.pad_sequences(ytrain, maxlen=maxlen,padding='post')
# y_test = sequence.pad_sequences(ytest, maxlen=maxlen,padding='post')
print('X_train shape:', Xtrain.shape)
print('X_test shape:', Xtest.shape)
# print('X_framtest shape:', Xframtest.shape)
# print('X_framtrain shape:', Xframtrain.shape)
print('y_train shape:', ytrain.shape)
print('y_test shape:', ytest.shape)

print('Build model...')
Xlenmax=0
for i in range(len(Xtrain)):
    for j in range(len(Xtrain[i])):
        Xlenmax=max(Xlenmax,Xtrain[i][j].shape[0])

f=open('accuracy.out','w')            
model = Sequential()                            
model.add(LSTM(512,return_sequences =True,input_dim=Xlenmax))     
model.add(TimeDistributedDense(output_dim=len(charac)))
model.add(Activation('softmax'))               
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
for iteration in range(1, 1000):
	print'Iteration', iteration       
	for i in range(Xtrain.shape[0]):
		if len(Xtrain[i])<=1:                                                              
			continue      
		Xt = sequence.pad_sequences(Xtrain[i], maxlen=Xlenmax,padding='post')
		model.fit( np.array([Xt,Xt]), np.array([ytrain[i],ytrain[i]]),nb_epoch=1,batch_size=2)
		count=0                                         
		totcount=0      
		for i in range(Xtest.shape[0]):
			if len(Xtest[i])<=1:             
				continue 
			Xt = sequence.pad_sequences(Xtest[i], maxlen=Xlenmax,padding='post')
			pred = model.predict(np.array([Xt,Xt]))
			for j in range(len(Xtest[i])):                                              
				if list(ytest[i][j]).index(1.0)==list(pred[0][j]).index(max(pred[0][j])):
					count+=1
				totcount+=1                                                   
		f.write("Iteration %d accuracy %f\n"%(iteration,count*1.0/totcount))
		f.flush()
f.close()
# for iteration in range(1, 60):
	# print'Iteration', iteration
# model.fit([X_train,Xframtrain], y_train, batch_size=batchsize, nb_epoch=100, validation_data=(X_test, y_test))
# score = model.evaluate(X_test, y_test)
# print('Test score:', score)
	# print('Test accuracy:', acc)
