#載入套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('kyphosis.csv')
print(df.head())
print(df.info())
#以seaborn pairplot快速掃瞄數字型資料有沒有趨勢
# sns.pairplot(df,hue='Kyphosis')
# plt.show()

#將資料分成訓練組及測試組
from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis',axis=1) #X資料集不需要Kyphosis資料
y = df['Kyphosis']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
#添加random_state，通過固定random_state的值，每次可以分割得到同樣訓練集和測試集

#使用決策樹演算法
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
#評估決策樹模型好壞
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#使用隨機森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100)
#從訓練組資料中建立隨機森林模型
rfc.fit(X_train,y_train)
#預測測試組的駝背是否發生
rfc_pred = rfc.predict(X_test)
#利用classification report來看precision、recall、f1-score、support
print(classification_report(y_test,rfc_pred))
#利用confusion matrix來看實際及預測的差異
print(confusion_matrix(y_test,rfc_pred))

