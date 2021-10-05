from sklearn import datasets
#load iris dataset
iris = datasets.load_iris()
#label species
print(iris.target_names)
#feature
print(iris.feature_names)
#iris data(top 5 record)
print(iris.data[0:5])
#iris label
print(iris.target)

#create Dataframe
import pandas as pd
df = pd.DataFrame({
    'sepal length': iris.data[:,0],
    'sepal width': iris.data[:,1],
    'petal length': iris.data[:,2],
    'petal width': iris.data[:,3],
    'species': iris.target
})

# print(df.head())

# Import train_test_split function
from sklearn.model_selection import train_test_split
X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = df['species']

#split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#import random forest model
from sklearn.ensemble import RandomForestClassifier
#create classifier
clf = RandomForestClassifier(n_estimators= 100)
#train model using training set
clf.fit(X_train,y_train)
#prediction
y_predict = clf.predict(X_test)
#model accuracy
from sklearn import metrics
acc = metrics.accuracy_score(y_test,y_predict)
print("Accuracy",acc)

# 丟資料進model分類
species_idx =clf.predict([[3,5,4,2]])[0] #y label值
print(iris.target_names[species_idx])

#finding important features
feature_imp = pd.Series(clf.feature_importances_,index = iris.feature_names)
print(feature_imp)

#bar plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x=feature_imp,y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#Generating the Model on Selected Features
'''We can remove the "sepal width" feature 
because it has very low importance '''
# X=data[['petal length', 'petal width','sepal length']]  # Removed feature "sepal length"
# y=data['species']                                       

#reference from https://www.kaggle.com/tcvieira/simple-random-forest-iris-dataset