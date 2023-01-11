import numpy as np
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import datasets


data=pd.read_csv("C:/Users/Sravanthi/Desktop/miniproject/parksrecmovies.csv")

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


# Spliting data into Feature and
X=data[['Borough','LocationType','day','time']]
y=data['turnoutStatus123']
print("transformed data")
print(X[0:5])
print(y[0:5])


from sklearn.model_selection import train_test_split
x_trainset,x_testset,y_trainset,y_testset=train_test_split(X,y,test_size=0.2,random_state=42)
tstree=DecisionTreeClassifier(criterion='entropy',max_depth=4)
tstree
tstree.fit(x_trainset,y_trainset)
predtree=tstree.predict(x_testset)
print(predtree[0:5])
print(y_testset[0:5])
from sklearn import metrics
import matplotlib.pyplot as plt
print("accuracy:",metrics.accuracy_score(y_testset,predtree))
fn=data.columns[1:5]
cn=data["turnoutStatus"].unique().tolist()
tstree.fit(X,y)
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
tree.plot_tree(tstree,feature_names=fn,class_names=cn,filled=True)
fig.savefig('prm1.png')
print(tree.plot_tree(tstree,feature_names=fn,class_names=cn,filled=True))
print(predtree)

#model validation
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

clf_tree = DecisionTreeClassifier();

clf_tree.fit(x_trainset, y_trainset); 

y_score1 = clf_tree.predict_proba(x_testset)[:,1]

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_testset, y_score1)

print('roc_auc_score for DecisionTree: ', roc_auc_score(y_testset, y_score1))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

