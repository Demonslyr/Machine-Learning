#First Attempt at the iris dataset. Looking for a classifier
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import random
import numpy as np
import tensorflow as tf

FLAGS = None
irisData = []

def importData():
    f = open(r"C:\Users\DrMur\DataSets\Iris\iris.data.txt", "r")
    data = []
    for line in f:
        data.append(line.rstrip('\n').split(','))
    return data

def getBatch(data, size):
    dataSamples = random.sample(data,size)
    batch_xs = []
    batch_ys = []
    for sample in dataSamples:
        batch_xs.append(sample[:len(sample)-1])
        batch_ys.append(getOneHot(sample[len(sample)-1]))
    return (batch_xs, batch_ys)

#purpose of code is tensorflow exp
def getOneHot(name):
    if name == "Iris-setosa":
        return [1,0,0]
    if name == "Iris-versicolor":
        return [0,1,0]
    if name == "Iris-virginica":
        return [0,0,1]

def main():
    #import my data probably using numpy?
    irisData = importData()
    random.shuffle(irisData)
    trainSplit = (len(irisData)//10)*6
    testSplit = len(irisData)-trainSplit
    irisTrain = irisData[:trainSplit]
    irisTest = irisData[:-testSplit]

    #create the model
    x = tf.placeholder(tf.float32, [None, 4])
    W = tf.Variable(tf.zeros([4,3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.matmul(x,W) + b

    #define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 3])

    #adding the more stable cross entropy
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)

    #now test the model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    desiredBatch = 25
    batchSize = desiredBatch if desiredBatch < trainSplit else trainSplit
    #train
    for i in range(10000):
        batch_xs, batch_ys = getBatch(irisTrain, batchSize )
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_xs, y_: batch_ys})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #hacky way to put the test data in tyhe form I need
    irisTest_x, irisTest_y_ = getBatch(irisTest, len(irisTest))

    print(sess.run(accuracy, feed_dict={
        x: irisTest_x, y_: irisTest_y_
    }))

if __name__ == '__main__':
    main()