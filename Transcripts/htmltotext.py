import os
import matplotlib.pyplot as plt
import math
import numpy as np
path='./html/'
path2='./text2/'
folder=os.listdir(path)
print folder
for x in folder:
	npath=os.path.join(path,x)
	if not(os.path.isdir(npath)):
		continue
	sfolder=os.listdir(npath)
	for y in sfolder:
		# print y
		if os.path.isdir(os.path.join(npath,y)):
			continue        
		f=open(os.path.join(npath,y),'r')
		a=f.read()
# fil=open('season01/0101.html','r')
# a=fil.read()
		b=a.split('<')
		# print b
		anew=[]
		for i in b:
			if len(i.split('>'))>1:
				anew.append(i.split('>')[1])
		# print anew
		anew = [z for z in anew if z!='\n']
		# print anew
		for i in range(len(anew)):
			# print anew[i]
			anew[i] = anew[i].replace('&#145;','\'')
			anew[i] = anew[i].replace('&nbsp;', '')
			anew[i] = anew[i].replace('&#146;', '\'')
			anew[i] = anew[i].replace('&quot;','"')
			anew[i] = anew[i].replace('&#133;','...')
			anew[i] = anew[i].replace('&amp','&')
			anew[i] = anew[i].replace('&#151;','-')
			anew[i] = anew[i].replace('&lt;','\'')

			# print anew[i]
		anew =''.join(anew)
		f1=open(os.path.join(os.path.join(path2,x),y[:-4] +'txt'),'w')
		for z in anew:
			f1.write(z) 
		f1.close()
