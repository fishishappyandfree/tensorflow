import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('data/',one_hot=True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

print(trainimg.shape)  # (55000,784)  the first parameter mean that how many samples,the sencond parameter mean that how many kinds feature
print(trainlabel.shape)  # (55000, 10)
print(testimg.shape)  # (10000, 784)
print(testlabel.shape)  # (10000, 10)

# because the later you need feed the batch data to the model,so in the model caculationg,the parameters should be uncentain

# so,the first step:
# the fllow vatiable x and y are feeded
x = tf.placeholder('float',[None,784],name='x')
y = tf.placeholder('float',[None,10],name='y')
W = tf.Variable(tf.random.normal((784,10), name='W'))
b = tf.Variable(tf.zeros([1]), name='b')

actv = tf.nn.softmax(tf.matmul(x,W)+b)

# cost = tf.reduce_mean(tf.reduce_sum(y*tf.log(actv)), name='cost')
# mean that take the loss about - p * log p, first we reduce the dimension to one dimension, then we can get the mean value by reduce_mean
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

pred = tf.equal(tf.argmax(actv,1),tf.argmax(y,1))  # by column find the max position

# to let the false or true to 0 and 1
accr = tf.reduce_mean(tf.cast(pred, 'float'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # to initial the parameter
    training_epochs = 50
    batch_size = 100
    display_size = 5

    # sess
    for epoch in range(training_epochs):
        avg_cost = 0
        num_batch = int(mnist.train.num_examples/batch_size)
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x:batch_xs, y:batch_ys})

            avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})/num_batch

        if training_epochs % display_size == 0:
            train_feeds = {x:batch_xs, y:batch_ys}
            test_feeds = {x:testimg, y:testlabel}
            train_arr = sess.run(accr,feed_dict=train_feeds)
            test_arr = sess.run(accr,feed_dict=test_feeds)
            print(sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))
            print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc:%.3f" % (epoch, training_epochs, avg_cost, train_arr, test_arr))



