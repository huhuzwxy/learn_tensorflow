import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

iris = datasets.load_iris()
#print(iris)
iris_dataframe = pd.DataFrame(data = np.c_[iris.data, iris.target], columns = np.append(iris.feature_names, 'labels'))
print(iris_dataframe)
print(iris_dataframe.isnull().sum())
print(iris_dataframe.groupby('labels').count())

x = iris_dataframe[iris.feature_names]
#print(x)
y = iris_dataframe.labels
#print(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 3)

knn = KNeighborsClassifier()
knn.fit(train_x, train_y)

pred_y = knn.predict(test_x)
print(metrics.accuracy_score(test_y, pred_y))

#data = pd.DataFrame([[1,2,3],[2,3,4]], columns = ['tree', 'flowers', 'vegetables'], index = ['a','b'])
#print(data)
#print(data.index)
#print(data.columns)

