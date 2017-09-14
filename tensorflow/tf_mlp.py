from __future__ import print_function

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_PATH = '/data/vision/fisher/data1/vsmolyakov/tensorflow/'

# Import MNIST data
mnist = input_data.read_data_sets("./MNIST-data/", one_hot=True)

# Parameters
lr_init = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

g1 = tf.Graph()
with g1.as_default():

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    pred = tf.matmul(layer_2, weights['out']) + biases['out']

    # learning rate schedule
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(lr_init, global_step, 100000, 0.96, staircase=True) 
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    # Initializing the variables
    init = tf.global_variables_initializer()

    #model checkpoint
    saver = tf.train.Saver()

    #tensorboard
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('learning rate', learning_rate)
    summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session(graph=g1) as sess:
    sess.run(init)

    patience_cnt = 0
    hist_loss = np.zeros(training_epochs)

    writer = tf.summary.FileWriter('./logs/mlp', graph=g1)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0. #reset avg cost per epoch
        total_batch = int(mnist.train.num_examples/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary = sess.run([optimizer, cost, summary_op],
                                     feed_dict={x: batch_x, y: batch_y})
            # compute average loss
            avg_cost += c / total_batch
            # write logs
            writer.add_summary(summary, epoch * total_batch + i)
        
        # Display logs per epoch step
        hist_loss[epoch] = avg_cost
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        # Save the model if cost improves
        if epoch > 0 and avg_cost <= min(hist_loss[:epoch]):
            print("loss improved, saving the model...")
            saver.save(sess, DATA_PATH + 'mlp/mlp_improvement.ckpt')

        # early stopping
        patience = 16
        min_delta = 0.01
        if epoch > 0 and hist_loss[epoch-1] - hist_loss[epoch] > min_delta:
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt > patience:
            print("early stopping...")
            break


    print("Optimization Finished!")
    print("saving final model...")
    saver.save(sess, DATA_PATH + 'mlp/mlp_final.ckpt')

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


with tf.Session(graph=g1) as sess2:
    print("restoring saved model...")
    saver.restore(sess2, DATA_PATH + 'mlp/mlp_final.ckpt')

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


#generate plots
plt.figure()
plt.plot(hist_loss, color='blue', alpha=0.8, label='Adam')
plt.title("MLP training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('./figures/mlp_loss_adam.png')




