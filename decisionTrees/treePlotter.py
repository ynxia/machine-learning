import matplotlib.pyplot as plt
# 定义文本框和箭头格式
# boxstyle表示文本框的类型，sawtooth表示锯齿形，fc表示边框线的粗细
decisionNode = dict(boxstyle="sawtooth",fc="0.8")
# 叶子节点的属性设置
leafNode = dict(boxstyle="round4",fc="0.8")
# 定义决策树的箭头属性
arrow_args = dict(arrowstyle="<-")

# 绘制带箭头的注解
# nodeTex表示要显示的文本，centerPt表示箭头头部的坐标，parentPt表示箭头尾部的坐标
# annotate表示在图像中添加注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # xy表示箭头尖的坐标
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
# 绘制图像
# def createPlot():
    # fig = plt.figure(1, facecolor='white') # 背景为白色
    # fig.clf() # 把画布清空
    # # createPlot.ax1表示图像的句柄，即用createPlot.ax1代表这个图像
    # # frameon表示是否绘制坐标轴矩形
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    # plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    # plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    # plt.show()

# 获取叶节点数量
def getNumLeafs(myTree):
    # 定义树的深度
    numLeafs = 0
    # 获得myTree的第一个键值，即第一个特征，分割的标签
    # 下面两行代码等价于firstStr = myTree.keys()[0]，下面的是python3的，这个是python2的
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    # 根据键值得到对应的值，即根据第一个特征分类的结果
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 如果secondDict[key]是一个字典，则递归计算叶子节点的个数，加到numLeafs上
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        # 否则该secondDict[key]为叶子节点
        else:   numLeafs +=1
    return numLeafs

# 获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    # 下面两行代码等价于firstStr = myTree.keys()[0]，下面的是python3的，这个是python2的
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

# 预先存储树的信息
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


# 在父子节点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

# 绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree) # 计算树的宽与高
    depth = getTreeDepth(myTree)
    # firstStr = myTree.keys()[0]为python2中的语法，此处应该使用python3的，和上文类似
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt) # 标记子节点属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

# 绘图，是最终的主函数，它调用了plotTree，plotTree又依次调用了前面几个函数
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()