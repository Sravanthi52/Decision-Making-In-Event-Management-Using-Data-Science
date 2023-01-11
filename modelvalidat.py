import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# Loading data
data = pd.read_csv("C:/Users/Sravanthi/Desktop/miniproject/parksrecmovies.csv")

# Import LabelEncoder
from sklearn import preprocessing

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
data['Borough']=le.fit_transform(data['Borough'])
data['LocationType']=le.fit_transform(data['LocationType'])
data['day']=le.fit_transform(data['day'])
#data['turnoutStatus']=le.fit_transform(data['turnoutStatus'])
data['time']=le.fit_transform(data['time'])

# Create feature and target arrays
X=data[['Borough','LocationType','day','time']]
y=data['turnoutStatus123']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
lw=2
for i in range(0,1):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
