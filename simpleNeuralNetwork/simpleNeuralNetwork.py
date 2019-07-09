import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels


print(trainimg.shape)  # (55000, 784)
print(trainlabel.shape)  # (55000, 10)
print(testimg.shape)  # (10000, 784)
print(testlabel.shape)  # (10000, 10)

# Network topologies
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

# input and outputs
x = tf.placeholder('float', [None,n_input])
y = tf.placeholder('float', [None,n_classes])

# network parameters
stddev = 0.1
weights = {
    'w1': tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))  # define the standard deviation of the normal distribution
}

bias = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1],stddev=stddev)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2],stddev=stddev)),
    'out':tf.Variable(tf.random_normal([n_classes],stddev=stddev))
}


def multilayer_perception(_X,_weights,_biases):
    # the out layer don't need to be activate function
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X,_weights['w1']),_biases['b1']))
    layer2 = tf.nn.sigmoid(tf.add(_biases['b2'],tf.matmul(layer_1,_weights['w2'])))
    return (tf.matmul(layer2,_weights['out'])+_biases['out'])


# prediction
pred = multilayer_perception(x,weights,bias)

# loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=pred))
optm = tf.train.GradientDescentOptimizer(0.01)
train = optm.minimize(cost)

corr = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
acrr = tf.reduce_mean(tf.cast(corr,'float'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_epochs = 20
    batch_size = 100
    dispaly_size = 4



    for epoch in range(train_epochs):
        ave_cost = 0.
        num_batch = int(mnist.train.num_examples / batch_size)
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed = {x:batch_xs, y:batch_ys}
            sess.run(train, feed_dict=feed)
            ave_cost += sess.run(cost, feed_dict=feed)/num_batch

        if epoch%dispaly_size == 0:
            train_feed = {x:batch_xs, y:batch_ys}
            test_feed= {x:testimg, y:testlabel}
            # train_accr = sess.run(acrr, feed_dict= train_feed)
            # test_accr = sess.run(acrr,feed_dict= test_feed)
            # print("Epoch: %03d/%03d cost:%.9f train_acc:%.3f test_acc:%.3f" % (epoch, train_epochs, ave_cost, train_accr,test_accr))
            print(type(batch_ys))