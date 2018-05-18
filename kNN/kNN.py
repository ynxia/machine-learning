from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','C','B']
    return group, labels

# inX是用于分类的输入向量，dataSet是训练集，labels是标签，k是近邻个数
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    # 计算距离
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #将inX变为二维数组，然后减去dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1) #numpy中axis=1表示沿行方向
    distances = sqDistances ** 0.5
    # 将距离从小到大排序
    sortedDistIndicies = distances.argsort()
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    # classCount.items()将classCount分解为元组列表
    # reverse表示降序排列
    # operator.itemgetter(1)表示获取对象第二个值，按照这个值的进行排序
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True) 
    return sortedClassCount[0][0]

# 将文本记录转换为numpy
def file2matrix(filename):
    # 读取文件行数
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    # 创建返回的numpy矩阵
    returnMat = zeros((numberOfLines, 3))   # line行3列
    classLabelVector = []
    index = 0
    # 解析数据文件到列表
    for line in arrayOfLines:
        line = line.strip()             # 移除首尾的空格/截取掉所有的回车符
        listFromLine = line.split('\t') # 通过指定分隔符对字符串进行切片
        returnMat[index,:] = listFromLine[0:3] # 选取前三个元素存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1])) # 将最后一列也加入到向量中，
        # 且我们需要告诉interpreter数据为整型，否则将自动当做字符串处理
        index += 1
    return returnMat,classLabelVector

# 数值归一化，使用min-max标准化（Min-Max Normalization）
def autoNorm(dataSet):
    minVals = dataSet.min(0) # axis=0; 每列的最小值，同理.min(1)则为每行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# 使用测试集对代码进行测试
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)   # 随机抽取10%的数据作为测试集
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

# 将图像转换为测试向量
def img2vector(filename):
    returnVect = zeros((1,1024)) # 32*32=1024
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 使用k近邻算法识别手写数字
def handwritingClassTest():
    hwlabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024)) # 矩阵的每行数据存储一个图像
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('.'))[0]
        hwLabels.append(classNumStr) # 从文件名记录下该图片的label
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))