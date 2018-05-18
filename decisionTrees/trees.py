from math import log
import operator
#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 保存数据集中实例的总数
    labelCounts = {}    # 保存为字典，因此应该使用花括号而不是方括号
    # 扩展：()表示元组tuple，[]表示列表list，{}表示字典dict
    # featVec表示dataSet中的一条数据
    for featVec in dataSet:
        currentLabel = featVec[-1] # 获取类别信息
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0 #将本来不在字典里的添加到字典里
        labelCounts[currentLabel] += 1 # 统计每个类别出现的次数
    shannonEnt = 0.0
    # prob表示每个类别出现的概率
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2) # 以2为底求对数
    return shannonEnt

# 创建简单数据集
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# 按照给定特征划分数据集
# 三个参数：待划分的数据集，划分数据集的特征以及特征的返回值
def splitDataSet(dataSet, axis, value):
    retDataSet = [] #以防修改了原始的数据集
    # featVec表示特征向量
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 最后一列记录类别标签
    baseEntropy = calcShannonEnt(dataSet)   # 计算整个数据集的原始香农熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        # 遍历所有特征
        featList = [example[i] for example in dataSet]# 创建一个列表保存这个特征下的所有实例
        uniqueVals = set(featList)       #对特征列表去重并保留结果
        # 【从列表中创建集合是python得到列表中唯一元素值的最快方法】
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #计算信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 得到出现次数最多的类别的名称
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树，参数：数据集和标签列表
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] # 包含了数据集的所有类标签
    if classList.count(classList[0]) == len(classList): # 递归函数的第一个停止条件：所有类标签完全相同
        return classList[0]
    if len(dataSet[0]) == 1: # 递归函数的第二个停止条件：使用完了所有特征，仍然不能将数据集划分为包含唯一类别的分组
        return majorityCnt(classList)
    # 下面开始创建树，使用字典类型储存类的信息
    # 当前数据集存储的最好的类别存储在bestFeat中
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) # 得到列表包含的所有属性值
    for value in uniqueVals:
        subLabels = labels[:] # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

# 使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr_1 = list(inputTree.keys())
    firstStr = firstStr_1[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    # Python3最好统一使用二进制方式读写，
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)