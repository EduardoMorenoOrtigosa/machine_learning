
##### 1. Reading the example dataset and data preparation - Iris dataset #####
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris() #Loading the dataset
print(iris.keys())

iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )

print(iris.head())

species = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')


iris['species'] = species

print(iris.describe())
print(iris.head())

##### 2. Training the model with full dataset #####
print(iris.columns)
X_train = iris[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
y_train = iris.species

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth = 3, random_state = 1)
model.fit(X_train,y_train)

##### 3. Creating the weights #####

import pickle
#pickle.dump(model, open('model_weights', "wb"))

##### 4. Studying possible outcome #####

print(y_train.unique())

X_test = X_train.sample(n=10)

print(X_test.head())
prediction = model.predict(X_test)

print(prediction)