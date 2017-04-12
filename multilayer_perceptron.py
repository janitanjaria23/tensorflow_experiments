"""
Wiki to multilayer perceptron:
https://www.wikiwand.com/en/Multilayer_perceptron
rectified function:
https://www.wikiwand.com/en/Rectifier_(neural_networks)
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_DATA/', one_hot=True)

learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1

n_hidden_1 = 256  # number of features in layer 1
n_hidden_2 = 256  # number of features in layer 2
n_input = 784  # 28 * 28 = 784
n_classes = 10  # 0-9 digits

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def multilayer_perceptron(x, weights, bias):
    """
    This is creating the model.takes input as the x and the weight and bias for all layers. Returns the outer layer result
    :param x:
    :param weights:
    :param bias:
    :return:
    """
    layer_1 = tf.add(tf.matmul(x, weights['h1']), bias['b1'])
    layer_1 = tf.nn.relu(layer_1)  # activation using rectifier function

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), bias['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.add(tf.matmul(layer_2, weights['out']), bias['out'])  # output layer has linear activation function
    return out_layer


weights = dict(h1=tf.Variable(tf.random_normal([n_input, n_hidden_1])),
               h2=tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
               out=tf.Variable(tf.random_normal([n_hidden_2, n_classes])))

biases = dict(b1=tf.Variable(tf.random_normal([n_hidden_1])),
              b2=tf.Variable(tf.random_normal([n_hidden_2])),
              out=tf.Variable(tf.random_normal([n_classes])))

pred = multilayer_perceptron(x, weights=weights, bias=biases)  # building the model

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimiser, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost:", \
                "{:.9f}".format(avg_cost)
    print "optimisation finished.."

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    print "correct_prediction: ", correct_prediction
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.images})
