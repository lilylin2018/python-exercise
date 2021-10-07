#import module
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

#load data
iris = datasets.load_iris()
X = iris.data
y = iris.target
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,random_state= 0)

#train model
clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
#SVC分類器
#kernel核函數(線性、多項式、高斯、sigmoid)
#C懲罰係數 越大代表錯誤容忍程度越低
#gamma 決定支援向量的多寡
clf.fit(X_train,y_train)

#predict
print(clf.predict(X_test))

#accuracy
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))