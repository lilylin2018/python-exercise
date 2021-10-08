from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pandas as pd

data = pd.read_csv("D:/python_exercise/SVM/UniversalBank.csv")
# print(data.head())

X = data.iloc[:,1:13].values
y = data.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.3, random_state= 0)
clf = SVC(kernel='rbf',random_state= 0)
#Radial basis function kernel 徑向基函數核
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

acc = cross_val_score(clf, X=X_train, y=y_train, cv= 10)
print(acc.mean())