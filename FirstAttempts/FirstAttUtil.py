import random

def importData():
    f = open(r"C:\Users\DrMur\DataSets\Iris\iris.data.txt", "r")
    data = []
    for line in f:
        data.append(line.rstrip('\n').split(','))
    return data

def getBatch(data, size, oneHot=False):
    dataSamples = random.sample(data,size)
    batch_xs = []
    batch_ys = []
    for sample in dataSamples:
        batch_xs.append(sample[:len(sample)-1])
        if(oneHot):
            batch_ys.append(getOneHot(sample[len(sample)-1]))
        else:
            batch_ys.append(getIntegerLabels(sample[len(sample)-1]))
    return (batch_xs, batch_ys)

#purpose of code is tensorflow exp
def getOneHot(name):
    if name == "Iris-setosa":
        return [1,0,0]
    if name == "Iris-versicolor":
        return [0,1,0]
    if name == "Iris-virginica":
        return [0,0,1]

def getIntegerLabels(name):
    if name == "Iris-setosa":
        return 0
    if name == "Iris-versicolor":
        return 1
    if name == "Iris-virginica":
        return 2