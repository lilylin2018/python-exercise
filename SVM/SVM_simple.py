#import module
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#隨意平面上的點
points = np.r_[[[-1,1],[1.5,1.5],[1.8,0.2],[0.8,0.7],[2.2,2.8],[2.5,3.5],[4,2]]]
#分成兩類
typeName = [0,0,0,0,1,1,1]

#建立模型
clf = svm.SVC(kernel='linear')
clf.fit(points,typeName)

#建立分類直線
sample = clf.coef_[0] #係數
slope = -sample[0]/sample[1] #斜率
lineX = np.arange(-2,5,1)
lineY = slope*lineX-(clf.intercept_[0])/sample[1]

#畫出劃分直線
plt.plot(lineX,lineY,color='blue',label='Classified Line')
plt.legend(loc='best') #繪製圖例
plt.scatter(points[:,0],points[:,1],c='Red')
plt.show()