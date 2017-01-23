# cmat = [[0 for i in range(8)] for j in range(7)]

# count=0
# totcount=0
# for i in range(X_test.shape[0]):
#     if len(X_test[i])<=1:
#         continue
#     pred = model.predict(np.array([X_test[i],X_test[i]]))
#     for j in range(len(X_test[i])):
#         cmat[list(y_test[i][j]).index(1.0)][list(pred[0][j]).index(max(pred[0][j]))]+=1
#         if list(y_test[i][j]).index(1.0)==list(pred[0][j]).index(max(pred[0][j])):
#             count+=1
#             totcount+=1




import os
import matplotlib.pyplot as plt
import math, pickle
import numpy as np
#from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix - 3 seasons')# ,without normalization

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, '.1f') ,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



cnf_mat=[[551, 53, 35, 39, 38, 29, 70],
 [24, 620, 38, 19, 46, 19, 101],
 [34, 20, 416, 37, 32, 21, 49],
 [30, 29, 52, 476, 36, 29, 60],
 [38, 59, 38, 38, 628, 30, 115],
 [26, 24, 45, 51, 42, 428, 58],
 [42, 62, 38, 22, 54, 23, 625]]

cnf_mat=np.array(cnf_mat)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_mat, classes=charac, title='Confusion matrix, without normalization', normalize=True)
#savefig('foo.png', bbox_inches='tight')
#plt.savefig('confusion.eps') 
#plt.close()
plt.show()