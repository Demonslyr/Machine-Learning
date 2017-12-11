#First Attempt at the iris dataset. Looking for a classifier
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

import FirstAttUtil as fu

FLAGS = None
irisData = []

def main():
    irisData = fu.importData()
    random.shuffle(irisData)
    trainSplit = (len(irisData)//10)*6
    testSplit = len(irisData)-trainSplit
    irisTrain = irisData[:trainSplit]
    irisTest = irisData[:-testSplit]

    #forrestParams
    num_epocs = 500 #epocs to train
    batch_size = 50 #samples per batch
    num_classes = 3 #3 iris
    num_features = 4 #4 features
    num_trees = 10 # number of trees
    max_nodes = 100 # maximum number of nodes

    #input and target data
    X = tf.placeholder(tf.float32, shape=[None, num_features])

    # For random forest, labels must be integers (the class id)
    Y = tf.placeholder(tf.int32, shape=[None])

    hparams = tensor_forest.ForestHParams(num_classes = num_classes,
                                            num_features=num_features,
                                            num_trees = num_trees,
                                            max_nodes=max_nodes).fill()

    #buildthe forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    #Get training graph and loss
    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)

    #measure accurracy
    infer_op = forest_graph.inference_graph(X)
    correct_prediction = tf.equal(tf.argmax(infer_op,1), tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables
    init_vars = tf.global_variables_initializer()

    #start tf session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars)

    # Training
    for i in range(1, num_epocs + 1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, batch_y = fu.getBatch(irisTrain, batch_size)
        _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        if i % 50 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

    # Test Model
    test_x, test_y = fu.getBatch(irisTest,len(irisTest))
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))


if __name__ == '__main__':
    main()
