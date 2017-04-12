# coding=utf-8
"""
Some insight into tensorflow graphs and sessions:
1. A graph defines the computation. It doesn’t compute anything, it doesn’t hold any values, it just defines the operations that you specified in your code.
2. A session allows to execute graphs or part of graphs. It allocates resources (on one or more machines) for that and holds the actual values of intermediate results and variables.
"""

import tensorflow as tf

hello = tf.constant('Hello TensorFlow!!!')

sess = tf.Session()

print sess.run(hello)
