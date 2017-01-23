import skipthoughts
import os                                          
import matplotlib.pyplot as plt
import math, pickle     
import numpy as np             
from sklearn import metrics                    
import itertools

model = skipthoughts.load_model()

# filepath = '../../skip-thoughts/skipvectorscenes.txt'
filepath = '../data/Friends/Transcripts/scenesvid.txt'

ps = pickle.load(open(filepath, "rb"))
psnew = [[] for i in range(len(ps)) ]  

X=[]
for i in range(len(ps)):
	for j in range(len(ps[i])):
		X.append(ps[i][j][1])
Xnew=[]
for i in range(len(X)):
    Xnew.append( ''.join([i if ord(i) < 128 else ' ' for i in X[i]]))

vectors=[]
for i in range(len(Xnew)):
	# print i, Xnew.index(i)
	if (Xnew[i].strip(' ')==''):
		Xnew[i]='a'
	# 	vectors.append(skipthoughts.encode(model, ['a']))
	# else:
	# 	vectors.append(skipthoughts.encode(model, [i]))
# vectors = skipthoughts.encode(model, Xnew)
vectors = skipthoughts.encode(model, Xnew)


count=0
for i in range(len(ps)):
	for j in range(len(ps[i])):
		psnew[i].append([ps[i][j][0],vectors[count]]+ps[i][j][2:])
		count+=1

pickle.dump(psnew, open('skipvectorscenesvid.txt', "wb"))
