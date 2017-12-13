#! python2
# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
                        #    random_state=0, shuffle=False)
from __future__ import print_function



import random
import FirstAttUtil as fu            


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
irisData = fu.importData()
random.shuffle(irisData)
trainSplit = (len(irisData)//10)*6
testSplit = len(irisData)-trainSplit
irisTrain = irisData[:trainSplit]
irisTest = irisData[:-testSplit]
batch_size = 50



batch_x, batch_y = fu.getBatch(irisTrain, batch_size)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(batch_x,batch_y)

print(clf.feature_importances_)

print(clf.predict([[6.1, 2.8, 4.0, 1.3]]))