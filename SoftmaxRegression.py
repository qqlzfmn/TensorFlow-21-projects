import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# disable debugging logs
tf.logging.set_verbosity(tf.logging.ERROR)  # tf.logging.WARN

# import data set
mnist = input_data.read_data_sets("data\MNIST", one_hot=True)

# set placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# set variables
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# set softmax classifier
y = tf.nn.softmax(tf.matmul(x, w) + b)

# set cross entropy loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
# set gradient descent optimizer
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# create session
sess = tf.InteractiveSession()
# initialize variables
tf.global_variables_initializer().run()

# 1000 times gradient descent
for _ in range(1000):
    train_x, train_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: train_x, y_: train_y})

# get and show results
test_x = mnist.test.images
test_y = mnist.test.labels
test_num_to_show = 3
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)), feed_dict={x: test_x, y_: test_y})
y_predict = sess.run(tf.argmax(y, 1)[:test_num_to_show], feed_dict={x: test_x, y_: test_y})
print(accuracy)
print(y_predict)
for i in range(test_num_to_show):
    plt.imshow(test_x[i].reshape(28, 28), 'gray')
    plt.show()
