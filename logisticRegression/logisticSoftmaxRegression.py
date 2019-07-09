import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('data/',one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

print(trainimg.shape)  # (55000, 784)
print(trainlabel.shape)  # (55000, 10)
print(testimg.shape)  # (10000, 784)
print(testlabel.shape)  # (10000, 10)

print(testlabel[0])

x = tf.placeholder("float", [None,784])
y = tf.placeholder("float", [None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros(10))

#Logistic regression model
# please pay attension to when run matrix multiply, the dimension must be adapred
actv = tf.nn.softmax(tf.matmul(x,W)+b)
#Cost function
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))
#Optimizer
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optm = optimizer.minimize(cost)

# print(actv.shape)  # (?, 10)

#Prediction
pred = tf.equal(tf.argmax(actv,1), tf.argmax(y,1))  # (？,10)  (?, 10)
#Accuracy
accr = tf.reduce_mean(tf.cast(pred, 'float'))
# Initializer
init = tf.global_variables_initializer()

# Session
with tf.Session() as sess:
    sess.run(init)

    # start to train data with logistic algrithm
    training_epochs = 50  # the times samples have been trained
    batch_size = 100  # every times how many data hava been trained
    display_size = 5  # when next size, print the prtdicted data

    # Mini-batch learning
    for epoch in range(training_epochs):
        avg_cost = 0.
        # to know that the number of the batch
        num_batch = int(mnist.train.num_examples/batch_size)
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x : batch_xs, y : batch_ys})
            feeds = {x : batch_xs, y : batch_ys}
            avg_cost += sess.run(cost, feed_dict=feeds)/num_batch

        # display
        if epoch % display_size == 0:
            feeds_train = {x:batch_xs, y:batch_ys}
            feeds_test = {x:mnist.test.images, y:mnist.test.labels}
            train_acc = sess.run(accr, feed_dict=feeds_train)
            test_acc = sess.run(accr,feed_dict=feeds_test)
            print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f" % (epoch,training_epochs,avg_cost,train_acc,test_acc))
print("DONE")
