import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Sravanthi/Desktop/miniproject/parksrecmovies.csv")
df.head()

df.dtypes

df.shape

df.describe()

df.info()

df.isnull().sum()

# histogram
sns.histplot(x='Attendance',y='day', data=df, )
plt.show()
sns.histplot(x='Attendance',y='time', data=df, )
plt.show()

# boxplot
sns.boxplot( x="Attendance", y='Borough', data=df, )
plt.show()

# scatter bivariate
sns.scatterplot( x="Attendance", y='Borough', data=df,
				hue='turnoutStatus', size='Attendance')

# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)

plt.show()

#boxplot outlier imputations
data=pd.read_csv("C:/Users/Sravanthi/Desktop/miniproject/parksrecmovies.csv")

sns.boxplot(data['Attendance'])
plt.title("Box Plot before imputation")
plt.show()

train = data.copy()
q1 = train['Attendance'].quantile(0.25)
q3 = train['Attendance'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
m = np.mean(train['Attendance'])
for i in train['Attendance']:
    if i > Upper_tail or i < Lower_tail:
            train['Attendance'] = train['Attendance'].replace(i, m)
sns.boxplot(train['Attendance'])
plt.title("Box Plot after mean imputation")
plt.show()

train = data.copy()
q1 = train['Attendance'].quantile(0.25)
q3 = train['Attendance'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
med = np.median(train['Attendance'])
for i in train['Attendance']:
    if i > Upper_tail or i < Lower_tail:
            train['Attendance'] = train['Attendance'].replace(i, med)
sns.boxplot(train['Attendance'])
plt.title("Box Plot after median imputation")
plt.show()

train = data.copy()
q1 = train['Attendance'].quantile(0.25)
q3 = train['Attendance'].quantile(0.75)
iqr = q3-q1
Lower_tail = q1 - 1.5 * iqr
Upper_tail = q3 + 1.5 * iqr
for i in train['Attendance']:
    if i > Upper_tail or i < Lower_tail:
            train['Attendance'] = train['Attendance'].replace(i, 0)
sns.boxplot(train['Attendance'])
plt.title("Box Plot after Zero value imputation")
plt.show()

print(train['Attendance'])

data['Attendance'] = data['Attendance'].replace(train['Attendance'])
  
# writing into the file
data.to_csv("C:/Users/Sravanthi/Desktop/miniproject/parksrecmovies.csv", index=False)
  
print(data)




