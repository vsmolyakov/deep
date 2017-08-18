import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import variance_scaling_initializer

import seaborn as sns
import matplotlib.pyplot as plt

def weight_variable(name, shape):
    he_normal = variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,
                                             seed=None, dtype=tf.float32) 
    return tf.get_variable(name, shape=shape, initializer=he_normal)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#dataset
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

#parameters
learning_rate = 0.001
training_epochs = 1000
batch_size = 50
display_step = 100

x = tf.placeholder(tf.float32, [None, 784])
y_= tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])

#layer 1
W_conv1 = weight_variable("W1", [5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#layer 2
W_conv2 = weight_variable("W2", [5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#layer 3
W_fc1 = weight_variable("W3", [7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#layer 5
W_fc2 = weight_variable("W5", [1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#optimizers
train_step1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step2 = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(cross_entropy)
train_step3 = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)
train_step4 = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess1:
    sess1.run(init)

    print "training with SGD optimizer..."
    avg_loss_sgd = np.zeros(training_epochs)
    for epoch in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, loss = sess1.run([train_step1, cross_entropy], 
                            feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        avg_loss_sgd[epoch] = loss

        if epoch % display_step == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 1.0})
            print("Epoch %04d, training accuracy %.2f"%(epoch, train_accuracy))
    #end for
    print("test accuracy %g" %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

with tf.Session() as sess2:
    sess2.run(init)

    print "training with momentum optimizer..."
    avg_loss_momentum = np.zeros(training_epochs)
    for epoch in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, loss = sess2.run([train_step2, cross_entropy], 
                            feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        avg_loss_momentum[epoch] = loss

        if epoch % display_step == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 1.0})
            print("Epoch %04d, training accuracy %.2f"%(epoch, train_accuracy))
    #end for
    print("test accuracy %g" %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

with tf.Session() as sess3:
    sess3.run(init)

    print "training with RMSProp optimizer..."
    avg_loss_rmsprop = np.zeros(training_epochs)
    for epoch in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, loss = sess3.run([train_step3, cross_entropy], 
                            feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        avg_loss_rmsprop[epoch] = loss

        if epoch % display_step == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 1.0})
            print("Epoch %04d, training accuracy %.2f"%(epoch, train_accuracy))
    #end for
    print("test accuracy %g" %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

with tf.Session() as sess4:
    sess4.run(init)

    print "training with Adam optimizer..."
    avg_loss_adam = np.zeros(training_epochs)
    for epoch in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, loss = sess4.run([train_step4, cross_entropy], 
                            feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        avg_loss_adam[epoch] = loss

        if epoch % display_step == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob: 1.0})
            print("Epoch %04d, training accuracy %.2f"%(epoch, train_accuracy))
    #end for
    print("test accuracy %g" %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


plt.figure()
plt.plot(avg_loss_sgd, color='blue', alpha=0.8, label='SGD')
plt.plot(avg_loss_momentum, color='red', alpha=0.8, label='Momentum')
plt.plot(avg_loss_rmsprop, color='green', alpha=0.8, label='RMSProp')
plt.plot(avg_loss_adam, color='black', alpha=0.8, label='Adam')
plt.title("Neural Network Optimizers")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()





















