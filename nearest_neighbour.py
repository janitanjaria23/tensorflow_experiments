import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

xtrain, ytrain = mnist.train.next_batch(5000)
xtest, ytest = mnist.test.next_batch(200)

xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# reduce_sum takes the sum of all elements across an axis in a tensor
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

pred = tf.arg_min(distance, 0)

accuracy = 0.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(xtest)):
        nearest_neighbour_index = sess.run(pred, feed_dict={xtr: xtrain, xte: xtest[i, :]})
        print "Test: %d " % i, "Prediction: %d" % (np.argmax(ytrain[nearest_neighbour_index])), "Actual class: %d" % (
            np.argmax(ytest[i]))

        if np.argmax(ytrain[nearest_neighbour_index]) == np.argmax(ytest[i]):
            accuracy += 1. / len(xtest)

    print "Completed.."
    print "The accuracy of the algorithm is: %.3f" % accuracy
