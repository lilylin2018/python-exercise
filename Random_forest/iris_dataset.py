'''
Iris dataset
古典花朵資料集，共150筆資料
attributes: sepal length, sepal width, petal length, petal width
labels: setosa山鳶尾, versicolor變色鳶尾, virginica維吉尼亞鳶尾
'''

# how to use
from sklearn import datasets
iris = datasets.load_iris()
#print(iris.DESCR) #資料集詳細描述

# attribute資料在iris.data中
# label 在iris.target中
print(iris.data[:2]) #第一筆資料&第二筆資料
print(iris.target[:2])

print(iris.feature_names)#特徵內容
print(iris.target_names)