import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

n_input = 784
n_output = 10
# the feature shape has 4 parameter, the first and second mean the height ,weight, the third mean that depth,the last mean the number of output channel(output feature graph)
weights = {
    # the feature layer
    'wc1': tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),  # a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    'wc2': tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
    # the full connect layer
    'wd1': tf.Variable(tf.random_normal([7*7*128,1024],stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([1024,n_output],stddev=0.1))
}

biaes = {
    'bc1': tf.Variable(tf.random_normal([64],stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([128],stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1024],stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output],stddev=0.1))
}


def conv_basic(_input, _w, _b, _keepratio):
    '''
    INPUT, for tensorflow, the image need to be the 4 dimension,shape=[batch_size, height, weight, channel]
    Computes a 2-D convolution given 4-D `input` and `filter` tensors.
    Given an input tensor of shape `[batch, in_height, in_width, in_channels]
    '''

    _input_r = tf.reshape(_input, shape=[-1,28,28,1])  # for shape, at most there are at most one parameter to -1,
    # Conv layer 1
    #For the most common case of the same horizontal and vertices strides, `strides = [1, stride, stride, 1]
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1,1,1,1], padding = 'SAME')

    # _mean, _variance = tf.nn.moments(_convl,[0,1,2])
    # _conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0,1,0.0001)

    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1,_b['bc1']))

    _pool1 = tf.nn.max_pool(_conv1,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)

    # CONV LAYER 2
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1,1,1,1], padding='SAME')

    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2,_b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # print(_pool_dr2.shape)
    #VECTORIZE
    _dense1 = tf.reshape(_pool_dr2, [-1,_w['wd1'].get_shape().as_list()[0]])
    #Fully Connected layer1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1,_w['wd1']),_b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    #Fully Connected Layer2
    _out = tf.add(tf.matmul(_fc_dr1,_w['wd2']), _b['bd2'])
    #Returns
    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1':_pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
           'fc1':_fc1, 'fc_dr1':_fc_dr1, 'out':_out}
    return out

# a = tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1))

x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float', [None,n_output])
keepratio = tf.placeholder('float')

#FUNCTIONS
_pred = conv_basic(x,weights,biaes,keepratio)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=_pred['out']))
optm = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
_corr = tf.equal(tf.argmax(_pred['out'],1),tf.argmax(y,1))
_accr = tf.reduce_mean(tf.cast(_corr,'float'))


# SAVE

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(a))

    training_epochs = 15
    batch_size = 16
    dispaly_size = 1
    for epoch in range(training_epochs):
        ave_cost = 1.0
        num_batch = 10
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x:batch_xs, y:batch_ys, keepratio:1.0}
            sess.run(optm,feed_dict={x:batch_xs, y:batch_ys, keepratio:0.7})
            ave_cost += sess.run(cost,feed_dict=feeds)/num_batch
        if epoch%dispaly_size == 0:
            train_feeds = {x:batch_xs, y:batch_ys, keepratio:1.0}
            # test_feeds = {x:testimg, y:testlabel, keepratio:1.0}
            train_accr = sess.run(_accr,feed_dict=train_feeds)
            # test_accr = sess.run(_accr,feed_dict=test_feeds)
            # print("Epoch: %03d/%03d cost: %.9f train_accr: %.3f test_accr: %.3f" % (epoch, training_epochs, ave_cost,train_accr, test_accr))
            print("Epoch: %03d/%03d cost: %.9f train_accr: %.3f" % (epoch, training_epochs, ave_cost, train_accr))