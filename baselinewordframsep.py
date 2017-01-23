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
Xfram=[]
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
        temXfram.append(x[j][1])
        temX.append(np.concatenate((onehotcharac(x[j][0][0]),x[j][0][1]),axis=0))
        if j==len(x)-1:
            temY.append(onehotcharac('end'))
        else:
            temY.append(onehotcharac(x[j+1][0][0]))
        # temY.append()
    Xfram.append(temXfram)
    X.append(temX)
    Y.append(temY)
print "Created X,Y"

Xtrain = np.array(X[:int(0.7*len(X))])
Xframtrain = np.array(Xfram[:int(0.7*len(X))])
ytrain = np.array(Y[:int(0.7*len(Y))])
Xtest = np.array(X[int(0.7*len(X)):])
Xframtest = np.array(Xfram[int(0.7*len(X)):])
ytest = np.array(Y[int(0.7*len(Y)):])

print(len(Xtrain), 'train sequences')
print(len(Xtest), 'test sequences')


maxlen=max([len(i) for i in data])
batchsize =32
max_features = 20000
Xlen = len(X[0][0])

print('Pad sequences (samples x time)')

print('X_train shape:', Xtrain.shape)
print('X_test shape:', Xtest.shape)
print('X_framtest shape:', Xframtest.shape)
print('X_framtrain shape:', Xframtrain.shape)
print('y_train shape:', ytrain.shape)
print('y_test shape:', ytest.shape)

Xlenmax=0
for i in range(len(Xfram)):
    for j in range(len(Xfram[i])):
        Xlenmax=max(Xlenmax,Xfram[i][j].shape[0])


print('Build model...')
modelfram = Sequential()
# modelfram.add(Embedding(output_dim=1000,return_sequences =True,input_shape=(maxlen, maxlenfram)))
modelfram.add(LSTM(1000,return_sequences =True,input_dim=maxlenfram))
modelfram.add(TimeDistributedDense(output_dim=500))#nb_filter=1000, filter_length= 3,
modeltext = Sequential()
# modeltext.add(Embedding(output_dim=1000,return_sequences =True,input_shape=(maxlen, Xlen)))
modeltext.add(LSTM(1000,return_sequences =True,input_dim=Xlen))
modeltext.add(TimeDistributedDense(output_dim=500))#nb_filter=1000, filter_length= 3,
merged = Merge([modeltext, modelfram], mode='concat')

model = Sequential()
# # model.add(Embedding(max_features, 512, dropout=0.2))
model.add(merged)

# model.add(LSTM(512,return_sequences =True,input_dim=Xlenmax))
# model.add(TimeDistributedDense(output_dim=len(charac)))
# model.add(Activation('softmax'))
model.add(LSTM(512,return_sequences =True,input_dim=1000))
model.add(TimeDistributedDense(output_dim=len(charac)))
model.add(Activation('softmax'))

# model.add(Dense(output_dim=len(charac)))
# model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

f=open('accuracynew.out','w')            
print Xlen,Xlenmax
print "Model compiled"
for iteration in range(1, 100):
    print'Iteration', iteration
    for i in range(Xtrain.shape[0]):
        if len(Xtrain[i])<=1:
            continue
        Xt = sequence.pad_sequences(Xframtrain[i], maxlen=Xlenmax,padding='post')
        # print np.array([Xtrain[i],Xtrain[i]]).shape,np.array([Xt,Xt]).shape
        model.fit([np.array([Xtrain[i],Xtrain[i]]),np.array([Xt,Xt])],np.array([ytrain[i],ytrain[i]]),batch_size=2,nb_epoch=1)
    count=0
    totcount=0
    for i in range(Xtest.shape[0]):
        if len(Xtest[i])<=1:
            continue
        Xt = sequence.pad_sequences(Xframtest[i], maxlen=Xlenmax,padding='post')
        pred = model.predict([np.array([Xtest[i],Xtest[i]]),np.array([Xt,Xt])])
        for j in range(len(Xtest[i])):
            if list(ytest[i][j]).index(1.0)==list(pred[0][j]).index(max(pred[0][j])):
                count+=1
            totcount+=1
    print "accuracy ",count*1.0/totcount
    f.write("Iteration %d accuracy %f\n"%(iteration,count*1.0/totcount))
    f.flush()
f.close()


# for iteration in range(1, 60):
    # print'Iteration', iteration
# model.fit([X_train,Xframtrain], y_train, batch_size=batchsize, nb_epoch=100, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test)
print('Test score:', score)
    # print('Test accuracy:', acc)
