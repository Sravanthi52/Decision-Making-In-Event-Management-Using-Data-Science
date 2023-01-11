import numpy as np
import pandas as pd

data=pd.read_csv("C:/Users/Sravanthi/Desktop/miniproject/parksrecmovies.csv")

data.head()

# Import LabelEncoder
from sklearn import preprocessing

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
data['Borough']=le.fit_transform(data['Borough'])
data['LocationType']=le.fit_transform(data['LocationType'])
#data['Event Name']=le.fit_transform(data['Event Name'])
data['day']=le.fit_transform(data['day'])
#data['turnoutStatus']=le.fit_transform(data['turnoutStatus'])
data['time']=le.fit_transform(data['time'])

# Spliting data into Feature and
X=data[['Borough','LocationType','day','time']]
y=data['turnoutStatus123']

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 70% training and 30% test

# Import MLPClassifer 
from sklearn.neural_network import MLPClassifier

# Create model object
clf = MLPClassifier(hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)

clf.fit(X_train,y_train)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Make prediction on test dataset
ypred=clf.predict(X_test)
print(ypred)

comparision = []
for i,j in zip(y_test,ypred):
    if i==j:
        comparision.append(True)
    else:
        comparision.append(False)
        
print(comparision)

if all(comparision):
    print('Both arrays are equal')
else:
    print('Both Arrays are not equal')
    

# Import accuracy score 
from sklearn.metrics import accuracy_score

# Calcuate accuracy
print("accuracy=",accuracy_score(y_test,ypred))

#model validation
from sklearn import datasets
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

clf_mlp = MLPClassifier(hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01);

clf_mlp.fit(X_train, y_train); 

y_score1 = clf_mlp.predict_proba(X_test)[:,1]

false_positive_rate3, true_positive_rate3, threshold1 = roc_curve(y_test, y_score1)

print('roc_auc_score for MLP classifier: ', roc_auc_score(y_test, y_score1))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - MLP classifier')
plt.plot(false_positive_rate3, true_positive_rate3)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
