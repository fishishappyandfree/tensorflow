import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data


mnist = input_data.read_data_sets('data/', one_hot=True)


# training_image = mnist.train.images
# training_label = mnist.train.labels
# test_image = mnist.test.images[:200]
# test_label = mnist.test.label[:200]



# visualize decoder setting
# parameters
learing_rate = 0.01
training_epochs = 5
batch_size = 256
display_step = 1
examples_to_show = 10


# network parameters
n_input = 784  # mnist data input  (img shape: 28 * 28)


# tf Graph input  (only pictures)
x = tf.placeholder('float', [None, n_input])


# hidden layer settings
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 128  # 2st layer num features
weights = {
    'encoder_h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2' : tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biaes = {
    'encoder_h1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_h2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_h1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_h2' : tf.Variable(tf.random_normal([n_input]))
}


# building the encoder
def encoder(x):
    # Encoder hidden layer with sigmoid activation #1
    layer1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x,weights['encoder_h1']),bias=biaes['encoder_h1']))

    # Encoder hidden layer with sigmoid activation #2
    layer2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(layer1,weights['encoder_h2']),bias=biaes['encoder_h2']))

    return layer2

# building the decode
def decoder(x):
    # Decoder hidden layer with sigmoid activation #1
    layer1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x,weights['decoder_h1']),bias=biaes['decoder_h1']))

    # Decoder hidden layer with sigmoid activation #2
    layer2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(layer1,weights['decoder_h2']),bias=biaes['decoder_h2']))
    return layer2


# construct model
encoder_op = encoder(x)
decoder_op = decoder(encoder_op)


# prediction
y_pred = decoder_op
# targets (labels) are the input data
y_true = x


# define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.square(y_pred-y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(cost)


# lauch the graph
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2019-03-25 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0])<1 :
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)

    # train cycle
    for epoch in range(training_epochs):
        ave_cost = 0.
        # loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # run optimization op (bachpropagation) and cost op (to get loss value)
            _, ave_cost = sess.run([optimizer, cost], feed_dict={x:batch_xs})
            ave_cost += ave_cost

            # display logs per epoch step
        if epoch % display_step == 0 :
            print('Epoch: %03d/%03d cost: %.9f' % (epoch,training_epochs,ave_cost))

    print('optimization finished')



    # applying encode and decode over test set
    encode_decode = sess.run(y_pred, feed_dict={x:mnist.test.images[:examples_to_show]})

    # compare original images with their reconstructions
    f, a = plt.subplots(2,10,figsize=(10,2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    plt.show()