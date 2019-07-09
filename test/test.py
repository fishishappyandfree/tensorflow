import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# after write the follow code,we can find that when writing the tensorflow,
# A: define the variable at first
# B: define object(also is method) by instance
# C:ds
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0,0.55)
    y1 = 0.3 * x1 + np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])

x1_data = []
y1_data = []
for i in vectors_set:
    x1_data.append(i[0])
    y1_data.append(i[1])
# plt.scatter(x1_data,y1_data,c='r',marker='.')
# plt.show()

W = tf.Variable(tf.random_uniform([1],-1.0,1.0),name='W')
b = tf.Variable(tf.zeros([1]),name='b')
y = W*x1_data+b

loss = tf.reduce_mean(np.square(y1_data-y),name='loss')

# we can find the GradientDescentOptimizer is class and minimize is a def
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss,name='train')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        sess.run(train)
        print('W = ',sess.run(W),'b = ',sess.run(b),'loss = ',sess.run(loss))

    plt.scatter(x1_data,y1_data,c='r',marker='.')
    plt.plot(x1_data,sess.run(W)*x1_data+sess.run(b))
    plt.show()