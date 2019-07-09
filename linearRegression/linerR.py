import numpy as np
import tensorflow as tf
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

# these are 1000 points radomly surrounded by the line about y=0.1x+0.3
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0,0.55)
    y1 = 0.1*x1+0.3+np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])

# to make some samples
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


#plt.figure(figsize=(30,40),dpi=20)
# plt.scatter(x_data,y_data,color='r')
# plt.show()

# to make one dimension matrix that the value is between -1 and 1
W = tf.Variable(tf.random_uniform([1],-1.0,1.0),name='W')
# to define the b matrix one dimension, the initial value is 0
b = tf.Variable(tf.zeros([1]),name='b')
# get the predicted value by calculating
y = W * x_data + b

# define the loss that the mean square error between the y and y_data
loss = tf.reduce_mean(tf.square(y-y_data),name='loss')
# to optimize the parameter by using gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.5)
# the progress training is minimum the error value
train = optimizer.minimize(loss,name='train')

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# value of the initializtion w and b
print('W=',sess.run(W),'b=',sess.run(b),'loss=',sess.run(loss))
# training 20 times
for step in range(20):
    sess.run(train)
    print('W=', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss))
writer = tf.summary.FileWriter('./tmp',sess.graph)

plt.scatter(x_data,y_data,c="r",marker='.')
plt.plot(x_data,sess.run(W)*x_data+sess.run(b))
plt.show()


import sys
print(sys.path)

