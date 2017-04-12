import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print "value of a: %d" % (sess.run(a)), "value of b: %d" % (sess.run(b))
    print "addition result: %d" % (sess.run(a + b))
    print "multiplication result: %d" % (sess.run(a * b))

a = tf.placeholder(tf.int16)  # This tensor will produce an error if evaluated. Its value must
# be fed using the `feed_dict` optional argument to `Session.run()`,
# `Tensor.eval()`, or `Operation.run()`.
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
multiply = tf.multiply(a, b)

with tf.Session() as sess:
    print "Addition results using placeholder values: %d" % (sess.run(add, feed_dict={a: 2, b: 3}))
    print "Multiplication results using placeholder values: %d" % (sess.run(multiply, feed_dict={a: 2, b: 3}))

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

matrix_mul_result = tf.matmul(matrix1,
                              matrix2)  # if too many values in the matrix are zero set the is_sparse flag = True

with tf.Session() as sess:
    print "Matrix multiplication result: %s" % (sess.run(matrix_mul_result))
