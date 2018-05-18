import kNN
from imp import reload
group, labels = kNN.createDataSet()
kNN.classify0([0,0],group,labels,3)

reload(kNN)
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')

import matplotlib
import matplotlib.pyplot as plt
from numpy import *
fig = plt.figure()
ax = fig.add_subplot(111) # 子图的一行一列第一块
# 等价于ax = fig.add_subplot(1,1,1)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()

reload(kNN)
normMat,ranges,minVals = kNN.autoNorm(datingDataMat)

reload(kNN)
kNN.datingClassTest()
