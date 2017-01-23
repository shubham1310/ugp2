import os
import matplotlib.pyplot as plt
import math, pickle
import numpy as np
fps = 23.98 
path='./textvid/'
folder=os.listdir(path)
characters =[]
maincharac=[]
othercharacters=[]
# manycharacters=[]
# closelist=[]
# notsure=[ "ross first has a look of 'huh' then changes it to sarcastic happy",'phoebe sr.','chandlers',]
# print folder
# containjoey=[]
# containross=[]
# containrach=[]
# containpho=[]
# containchan=[]
# containmon=[]
scene=[]
occurence={}
episodescene=[]
numrejected=0
def updateoccurence(couldbena):
	if couldbena in occurence:
		occurence[couldbena]+=1
	else:
		# print y, couldbename, b.index(i)
		occurence[couldbena]=1

def charactercount(key):
	return key.count('joey') + key.count('ross') + key.count('chan') + key.count('monica') + key.count('mnca') + key.count('rach') + key.count('phoebe')


subpath = '../subs/'
folder=os.listdir(subpath)
st='Friends.S'
files={}
for x in folder:
	npath=os.path.join(subpath,x)
	if not(os.path.isdir(npath)):
		continue
	sfolder=os.listdir(npath)
	for y in sfolder:
		if os.path.isdir(os.path.join(npath,y)) or y[0] =='.':
			continue
		season= y[len(st): len(st)+2]
		episode = y[len(st)+3: len(st)+5]
		text = open(os.path.join(npath,y),'r').read().replace('\r','').split('\n\n')
		refined=[]
		for i in text:
			j=i
			if j.strip(' ').strip('\n') == '':
				continue
			# print j
			while not(j[0].isdigit()):
				j=j[1:]
			if (j.strip(' ') == ''):
				continue
			seg = j.split('\n')
			x=seg[1].split(' --> ')
			temp=x[0].split(':')   
			temp2=temp[2].split(',')       
			val1=int((int(temp[0])*3600+int(temp[1])*60+int(temp2[0])+int(temp2[1])*0.001)*fps)
			temp=x[1].split(':')   
			temp2=temp[2].split(',')       
			val2=int((int(temp[0])*3600+int(temp[1])*60+int(temp2[0])+int(temp2[1])*0.001)*fps)
			if len(seg) > 3:
				if seg[2][0]=='-':
					refined.append([seg[2][1:],val1,(val1*len(seg[3])+val2*len(seg[2]))/(len(seg[2])+len(seg[3]))])
					refined.append([seg[3][1:],(val1*len(seg[3])+val2*len(seg[2]))/(len(seg[2])+len(seg[3])),val2])
				else:
					refined.append([seg[2]+ ' ' +seg[3],val1,val2])
			else:
				refined.append([seg[2],val1,val2])
		files[season+episode] = refined

def listinter(l1,l2):
	mark = [0 for i in range(len(l2))]
	count =0
	for i in l1:
		for j in range(len(l2)):
			if mark[j] ==0 and l2[j]==i:
				mark[j]=1
				count+=1
	return count

def cleanu(l1):
	l=[]
	for i in l1:
		j=i
		j=j.replace('.',' ')
		j=j.replace('!',' ')
		j=j.replace('-',' ')
		j=j.replace('?',' ')
		j=j.replace('[',' ')
		j=j.replace(']',' ')
		j=j.replace('(',' ')
		j=j.replace(')',' ')
		j=j.replace('  ', ' ')
		l.append(j)
	return l

folder=os.listdir(path)
numross=0
for x in folder:
	npath=os.path.join(path,x)
	if not(os.path.isdir(npath)):
		continue
	sfolder=os.listdir(npath)
	for y in sfolder:
		# print y
		if os.path.isdir(os.path.join(npath,y)) or y[0] =='.':
			continue
		# print y
		f=open(os.path.join(npath,y),'r')
		a=f.read()
		a=a.replace('\r','')
		subtitlenum=0
		flag=True
		numscene=0
		b=a.split('\n')
		# print b[:10]
		tempscene=[]
		countscene=True
		for i in b:
			if flag and '[' in i :
				# numscene+=1
				flag=False
				continue
			if 'scene' in i.lower():
				if countscene and not(tempscene==[]):
					scene.append(tempscene)
				else:
					# print "rejected"
					# print tempscene
					numrejected+=1
				numscene+=1
				tempscene=[]
				countscene=True
			if not(flag) and len(i.split(':'))>1 and not('[' in i) :
				# print "I conTRIBUTED"
				couldbename=i.split(':')[0].lower()
				if charactercount(couldbename)>1:
					while '(' in couldbename and  ')' in couldbename and charactercount(couldbename)>1:
						# print couldbename
						couldbename = couldbename[:couldbename.find('(')]+couldbename[couldbename.find(')')+1:]
						couldbename =couldbename.strip(' ')
					# print couldbename
					if '(' in couldbename or ')' in couldbename :
						# print 'I continued'
						continue
					couldbename =couldbename.strip(' ')


					
					# dialog = i.split(':')[1]
					# ma = 1
					# print ma
					# val =len(files[y[:-4]])-10
					# for j in range(max(min(len(files[y[:-4]])-40,subtitlenum-40),0),min(subtitlenum+40, len(files[y[:-4]]))):
					# 	intersec =listinter(dialog.lower().split(' '), files[y[:-4]][j][0].lower().split(' '))
					# 	# print dialog.lower().split(' ')
					# 	# print files[y[:-4]][j][0].lower().split(' ')
					# 	# print intersec
					# 	if intersec > ma:
					# 		# print "has been here"
					# 		val=j
					# 		ma = intersec 
					# print "\n\nmatched for actual"
					# print "from subtitle :",files[y[:-4]][val][0].lower()
					# print "from transcript :", dialog.lower()
					# subtitlenum=val


					if charactercount(couldbename)>1:
						# print "rejected ", couldbename
						countscene=False
						othercharacters.append(couldbename)
						updateoccurence(couldbename)
						continue
				couldbename =couldbename.strip(' ')

				dialog = i.split(':')[1]
				
				ma = 0
				# print ma
				# val =len(files[y[:-4]])-10
				for j in range(max(min(len(files[y[:-4]])-40,subtitlenum-40),0),min(subtitlenum+40, len(files[y[:-4]]))):
					intersec =listinter(cleanu(dialog.lower().split(' ')), cleanu(files[y[:-4]][j][0].lower().split(' ')))
					# print dialog.lower().split(' ')
					# print files[y[:-4]][j][0].lower().split(' ')
					# print intersec
					if intersec > ma:
						# print "has been here"
						val=j
						ma = intersec 
				print "\n\nmatched for actual"
				print "from subtitle :",files[y[:-4]][val][0].lower()
				print "from transcript :", dialog.lower()
				subtitlenum=val


				if 'joey' == couldbename:
					# containjoey.append('joey' )    
					tempscene.append(['joey',i.split(':')[1].lower(),y,files[y[:-4]][val][1],files[y[:-4]][val][2]])
					maincharac.append('joey')
				elif 'ross' == couldbename:
					# containross.append(couldbename)
					numross+=1
					tempscene.append(['ross',i.split(':')[1].lower(),y,files[y[:-4]][val][1],files[y[:-4]][val][2]])
					maincharac.append('ross')
				elif 'chandler' == couldbename:
					tempscene.append(['chandler',i.split(':')[1].lower(),y,files[y[:-4]][val][1],files[y[:-4]][val][2]])
					# containchan.append(couldbename)
					maincharac.append('chandler')
				elif 'monica' == couldbename or 'mnca' == couldbename :
					tempscene.append(['monica',i.split(':')[1].lower(),y,files[y[:-4]][val][1],files[y[:-4]][val][2]])
					# containmon.append(couldbename)
					maincharac.append('monica')
				elif 'rachel' == couldbename or 'rach'==couldbename:
					# containrach.append(couldbename)
					tempscene.append(['rachel',i.split(':')[1].lower(),y,files[y[:-4]][val][1],files[y[:-4]][val][2]])
					maincharac.append('rachel')
				elif 'phoebe' == couldbename:
					# containpho.append(couldbename)
					tempscene.append(['phoebe',i.split(':')[1].lower(),y,files[y[:-4]][val][1],files[y[:-4]][val][2]])
					maincharac.append(couldbename)
				elif ('joey' in couldbename) or ('ross' in couldbename) or ('chan' in couldbename) or ( 'monica' in couldbename or 'mnca' in couldbename) or('rach' in couldbename) or('phoebe' in couldbename):
					tempscene.append(['other',i.split(':')[1].lower(),y,files[y[:-4]][val][1],files[y[:-4]][val][2]])
					characters.append(couldbename)
				else:
					tempscene.append(['other',i.split(':')[1].lower(),y,files[y[:-4]][val][1],files[y[:-4]][val][2]])
					othercharacters.append(couldbename)
				# print "there"

				updateoccurence(couldbename)
		if countscene and not(tempscene==[]):
			scene.append(tempscene)
		else:
			numrejected+=1
		episodescene.append([y, numscene])

pickle.dump(scene, open('scenesvid.txt', "wb"))


print "\n\n\n\n================Number of final dialoges per person================="
charac=['joey','ross','chandler','monica','rachel','phoebe','other']
numdia=[0 for i in range(len(charac))]
for i in scene:
	# print i
	for j in i:
		# print j
		# print "priting charac",j[0], charac[charac.index(j[0])]
		numdia[charac.index(j[0])]+=1
for i in range(len(charac)):
	print charac[i],numdia[i]
# print "number of dialogues for ross : ", numross

characters= list(set(characters))
maincharac= list(set(maincharac))
othercharacters = list(set(othercharacters))
# for i in characters:
# 	if '#' in i:
# 		othercharacters.append(i)
# for i in othercharacters:
# 	characters.remove(i)
# nummorethanone=0
# for key,value in occurence.iteritems():
# 	if charactercount(key)>1:
# 		nummorethanone+=value
# print scene
print "Number of total scene accepted: ",len(scene)
print "Total number of scene rejected: ",numrejected
print "Average number of dialogue: %.5f"%(sum([len(x) for x in scene])*1.0/len([len(x) for x in scene]))

print "\n\n\n\n================Number of scenes per episode================="
episodescene =sorted(episodescene,key = lambda x: x[1])
for x in episodescene:
		print x[0],x[1]
print "\n\n=============Set of Main characters==========="
print maincharac
print "\n\n=============Set of characters containing the main characters==========="
print characters
print "\n\n=============Set of Other characters==========="
print othercharacters
# print containjoey
print "\n\n=============Number of Main characters==========="
print len(maincharac) 
print "\n\n=============Occurence of characters==========="

for key,value in sorted(occurence.iteritems(), key=lambda (k,v): (v,k)):
	print key+ '  ::::  ' + str(value)

# print "\nNumber of more than one occurence: %d"%(nummorethanone)
print "\nTotal number of dialogues: %d"%(sum(occurence.values()))