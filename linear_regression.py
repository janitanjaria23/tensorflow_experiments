import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random
learning_rate = 0.01
training_epocs = 1000
display_step = 50

train_x = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
print train_x

train_y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

print train_y

x = tf.placeholder("float")
y = tf.placeholder("float")

n_samples = train_x.shape[0]
print n_samples

w = tf.Variable(rng.rand(), name="weight")  # automatically creates these variables as global variables
b = tf.Variable(rng.rand(), name="bias")

pred = tf.add(tf.multiply(x, w), b)

cost = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * n_samples)  # mean squared error
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epocs):
        for (x1, y1) in zip(train_x, train_y):
            sess.run(optimiser, feed_dict={x: x1, y: y1})

        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={x: train_x, y: train_y})
            print "Epoch:", '%04d' % (epoch + 1), "cost", "{:.9f}".format(c), \
                "W:", sess.run(w), "b:", sess.run(b)

    print "Optimisation finished.."
    training_cost = sess.run(cost, feed_dict={x: train_x, y: train_y})
    print "Training Cost: ", training_cost, "W: ", sess.run(w), "b: ", sess.run(b)

    plt.plot(train_x, train_y, "ro", label="Original Data")
    plt.plot(train_x, sess.run(w)*train_x + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()
    print plt
    plt.savefig("linear_regression_plot.png")

