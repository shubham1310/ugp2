import os
import matplotlib.pyplot as plt
import math
import numpy as np
path='.'
folder=os.listdir(path)
st=[]
for x in folder:
    npath=os.path.join(path,x)
    if not(os.path.isdir(npath)):
        continue
    sfolder=os.listdir(npath)
    for y in sfolder:
        if os.path.isdir(os.path.join(npath,y)):
            continue
        print y
        flag=True
        stn=[]
        f=open(os.path.join(npath,y),'r')
        a=f.read()
        b=a.split('\n')
        for i in range(len(b)-5):
            if b[i]=='' or b[i]=='\r':
                y=b[i+2] 
            elif flag:
                y=b[i+1]
            if b[i]=='' or b[i]=='\r' or flag:
                if not(' --> ' in y):
                    continue
                x=y.split(' --> ')
                # print x
                if(flag):         
                    flag=False
                else:
                    temp=x[0].split(':')   
                    temp2=temp[2].split(',')       
                    val=int(temp[0])*3600+int(temp[1])*60+int(temp2[0])+int(temp2[1])*0.001
                    st.append(val-preval)
                    stn.append(val-preval)
                    # print val,preval
                    # print x
                temp=(x[1].split('\r'))[0].split(':')   
                temp2=temp[2].split(',')       
                preval=int(temp[0])*3600+int(temp[1])*60+int(temp2[0])+int(temp2[1])*0.001
        print [ round(i,2) for i in stn]
st=[int(round(i)) for i in st ]
# print st
plt.hist(st, bins=np.arange(min(st), 20))
# plt.hist(range(0,max(st)), st)
plt.show()