# Import necessary modules
import numpy as np
import pandas as pd
import sklearn
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
y=data['turnoutStatus']


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size = 0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print(knn.predict(X_test))

from sklearn import metrics
import matplotlib.pyplot as plt
print("accuracy:",metrics.accuracy_score(y_test,knn.predict(X_test)))
##
###model validation
##from sklearn import datasets
##from sklearn.metrics import roc_curve, roc_auc_score
##from sklearn.model_selection import train_test_split
##import matplotlib.pyplot as plt
##
##clf_knn = KNeighborsClassifier(n_neighbors=7);
##
##clf_knn.fit(X_train, y_train); 
##
##y_score1 = clf_knn.predict_proba(X_test)[:,1]
##
##false_positive_rate2, true_positive_rate2, threshold1 = roc_curve(y_test, y_score1)
##
##print('roc_auc_score for KNN model: ', roc_auc_score(y_test, y_score1))
##
##plt.subplots(1, figsize=(10,10))
##plt.title('Receiver Operating Characteristic - KNN model')
##plt.plot(false_positive_rate2, true_positive_rate2)
##plt.plot([0, 1], ls="--")
##plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
##plt.ylabel('True Positive Rate')
##plt.xlabel('False Positive Rate')
##plt.show()

#predict
datran=pd.read_csv("C:/Users/Sravanthi/Desktop/miniproject/prmtransform.csv")
print(datran)
print("enter event parameters")

b=int(input("borough:"))
lt=int(input("location type:"))
d=int(input("day:"))
t=int(input("time:"))

x_new=[[b,lt,d,t]]
pred=knn.predict(x_new)
print("predicted attendance=",pred)
    

