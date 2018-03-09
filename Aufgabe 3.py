from scipy.io import arff
import numpy as np
from sklearn.naive_bayes import GaussianNB
from operator import itemgetter


def Feature_Cut(Data, Feature_To_Cut):
    Data1 = Data.copy()
    names = list(Data1.dtype.names)
    names.remove(Feature_To_Cut)
    Data2 = Data1[names]
    # print(Data2[0])
    # 3 print(names)
    # print(Data1[0])
    return Data2
# cut the meaningless feature


def Change_Category(data, meta):
    Data = data.copy()

    for i in range(15):
        Array = []
        m = 0
        if meta.types()[i] == 'nominal':
            # print(i)
            for vektor in Data:
                if vektor[i] not in Array:
                    Array.extend([vektor[i]])
                    vektor[i] = m
                    m = m + 1
                else:
                    vektor[i] = Array.index(vektor[i])
    return Data
#change all of the attribution type to number

def stratifiedSampling(data: list):
    length = len(data[0])
    samplingData = []
    unlabeledData = data.copy()
    listOfSampleAndUnlabled = []
    numOfClass1 = 0
    numOfClass2 = 0
    numOfClass1Sampled = 0
    numOfClass2Sampled = 0
    dataLen = len(data)
    restLen = len(data)
    i = 0
    for line in data:
        if line[length - 1] == 1:
            numOfClass1 = numOfClass1 + 1
    for line in data:
        if line[length - 1] == 0:
            numOfClass2 = numOfClass2 + 1

    print(numOfClass1, numOfClass2)
    while i <= dataLen / 10:
        ran = np.random.randint(restLen - 1)
        if unlabeledData[ran][length - 1] == 1:
            if numOfClass1Sampled < numOfClass1 / 10:
                samplingData.append(unlabeledData[ran])
                del unlabeledData[ran]
                restLen = restLen - 1
                numOfClass1Sampled = numOfClass1Sampled + 1
                i = i + 1
        if unlabeledData[ran][length - 1] == 0:
            if numOfClass2Sampled < numOfClass2 / 10:
                samplingData.append(unlabeledData[ran])
                del unlabeledData[ran]
                restLen = restLen - 1
                i = i + 1
                numOfClass2Sampled = numOfClass2Sampled + 1
    listOfSampleAndUnlabled.append(samplingData)
    listOfSampleAndUnlabled.append(unlabeledData)
    return listOfSampleAndUnlabled
    # return the array of sapmled data'listOfSampleAndUnlabled[0]' and unlabeled data'listOfSampleAndUnlabled[1]'


def ChangeTypeToInt(Data, meta):
    length = len(Data[0])
    wide = len(Data)
    newData = np.zeros((wide, length))
    for i in range(wide):
        for j in range(length):
        #    print(Data[i][j])
            if meta.types()[j] == 'nominal':
                newData[i][j] = int(Data[i][j])
            else:
                newData[i][j] = Data[i][j]
         #   print(newData[i][j])

    newData = newData.tolist()
    return newData
    # return a list of digitalied Data


data, meta = arff.loadarff('census-income.arff')

Data = Change_Category(data, meta).copy()




newData = ChangeTypeToInt(Data, meta)
labelofSample = []
featureofsample=[]
tureLabel=[]
samplingData = stratifiedSampling(newData)[0].copy()
unlabeledData = stratifiedSampling(newData)[1].copy()

testData=unlabeledData.copy()
numOfTestcase=len(testData)
for line in samplingData:
    labelofSample.append(line[14])
    featureofsample.append(list(line[:14]))

for line in unlabeledData:
    tureLabel.append(line[-1])
    #print(len(line))
    line.append(0)
    line.append(0)
    line.append(0)


print(len(testData[1]))
#featureofsample = samplingData.copy()
numOfInter=0
while len(unlabeledData) > 0:
    numOfInter=numOfInter+1
    # m = []
    # for line in featureofsample:
    #     m.append(list(line[:14]))

    X = np.array(featureofsample)
    Y = np.array(labelofSample)
    gnb = GaussianNB()
    p = gnb.fit(X, Y)

    for line in unlabeledData:
        #print(len(line))
        proba=gnb.predict_proba([line[:14]])
        line[15] = (gnb.predict([line[:14]])[0])
        line[16] = (proba[0][0])
        line[17]=(proba[0][1])
        #print(line)

    numOfCorrect=0
    for line in testData:
        #print(line[14],gnb.predict([line[:14]]))
        if line[14]==(gnb.predict([line[:14]])):
            numOfCorrect=numOfCorrect+1


    accuracy=numOfCorrect/numOfTestcase
    print('interaction',numOfInter,'Accuracy=',accuracy)
    #print('jjjjj',gnb.score(testData,tureLabel))
    #print(unlabeledData)
    unlabeledData = sorted(unlabeledData, key=itemgetter(-2),reverse=True)
    #print(unlabeledData)
    i=0
    for line in unlabeledData:
        print(line)
        if i<500 :
            labelofSample.append(line[15])#####
            featureofsample.append(line[:14])
            #print(line)
            del line
            i=i+1

    unlabeledData = sorted(unlabeledData, key=itemgetter(-1),reverse=True)
    #print(unlabeledData)
    j=0
    for line in unlabeledData:
        if j<200:
            labelofSample.append(line[15])####
            featureofsample.append(line[:14])
            del line
            j=j+1
    print('numberofsam',len(featureofsample))
    # print(unlabeledData)
# print(unlabeledData)



