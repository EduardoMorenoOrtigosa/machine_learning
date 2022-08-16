

##### 1. Reading dataset - Credit dataset #####
from sklearn import datasets
import pandas as pd
import numpy as np

data = pd.read_csv("bank.csv")
data = data.dropna()

print(data.shape)
print(list(data.columns))

'''
Input variables

loan_applicant_id (numeric)
age (numeric)
education : level of education (categorical)
years_with_current_employer (numeric)
years_at_current_address (numeric)
household_income: in thousands of USD (numeric)
debt_to_income_ratio: in percent (numeric)
credit_card_debt: in thousands of USD (numeric)
other_debt: in thousands of USD (numeric)

Predict variable (desired target):

y — has the loan applicant defaulted on his loan? (binary: “1”, means “Yes”, “0” means “No”)
'''

print(data["education"].unique())

##### 2 - Data preparation

# tranform cat variables
cat_vars=["education"]

for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1 = data.join(cat_list)
    data = data1

cat_vars = ["education"]
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]

data_final = data[to_keep]
data_final.drop(['loan_applicant_id'], axis=1, inplace=True)
print(data_final.columns.values)

# oversamplijng minority class - default = 1 (yes)

X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

os = SMOTE(random_state=12345)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12345)

columns = X_train.columns
X, y = os.fit_resample(X_train, y_train)
X = pd.DataFrame(data=X,columns=columns )
y = pd.DataFrame(data=y,columns=['y'])

print("Length of oversampled data is ",len(X))
print("Number of no default in oversampled data ",len(y[y['y']==0]))
print("Number of default ",len(y[y['y']==1]))
print("Proportion of no default data in oversampled data is ",len(y[y['y']==0])/len(X))
print("Proportion of default data in oversampled data is ",len(y[y['y']==1])/len(X))

##### 3 - Training the model #####

### 1. p-value selection ###
import statsmodels.api as sm

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

pvalue = pd.DataFrame(result.pvalues,columns={'p_value'},)
print(pvalue)

pvs=[]
# keep only those which pvalue less than 0.05 
for i in range (0, len(pvalue["p_value"])):
    if pvalue["p_value"][i] < 0.05:
        pvs.append(pvalue.index[i])

# remove B0
if 'const' in pvs:
    pvs.remove('const')
else:
    pvs 

print(pvs)
print(len(pvs))

# retrain the model to check significance again
X = X[pvs]
y = y['y']
logit_model = sm.Logit(y,X)
result = logit_model.fit()
print(result.summary())


### 2. Model development ###
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12345)
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)

### 3. Model evaluation ###

from sklearn.metrics import accuracy_score
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
print(logit_roc_auc)

##### 4 - Generating weights for model deployment #####

final_logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
final_logreg.fit(X, y)

import pickle
pickle.dump(final_logreg, open('model_weights_pd.pkl', "wb"))

print(X.info())
print(X.columns)
