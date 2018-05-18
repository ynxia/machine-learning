#打开cmd首先输入的信息：
#cd Desktop\trees
#python trees_run.py

import trees
from imp import reload
reload(trees)
myDat,labels = trees.createDataSet()
print(myDat)
print(trees.calcShannonEnt(myDat))

reload(trees)
myDat,labels = trees.createDataSet()
print(myDat)
print(trees.splitDataSet(myDat,0,1))
print(trees.splitDataSet(myDat,0,0))

reload(trees)
myDat,labels = trees.createDataSet()
myTree = trees.createTree(myDat,labels)
print(myTree)

import treePlotter

reload(treePlotter)
treePlotter.retrieveTree(1)
myTree = treePlotter.retrieveTree(0)
print(treePlotter.getNumLeafs(myTree))
print(treePlotter.getTreeDepth(myTree))

reload(treePlotter)
myTree = treePlotter.retrieveTree(0)
myTree['no surfacing'][3] = 'maybe'
print(myTree)
treePlotter.createPlot(myTree)

myDat,labels = trees.createDataSet()
print(labels)
myTree = treePlotter.retrieveTree(0)
print(myTree)
trees.classify(myTree,labels,[1,0])
trees.classify(myTree,labels,[1,1])

trees.storeTree(myTree,'classifierStorage.txt')
trees.grabTree('classifierStorage.txt')

# 预测隐形眼镜类型
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lenseLabels=['age','prescript','astigmatic','tearRate']
lensesTree=trees.createTree(lenses,lenseLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)



