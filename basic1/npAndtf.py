import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# to practice the variable showed
# # float32
# #tf.zeros([3,4],int32) =>
# x = tf.linspace(10.0,12.0,3,name="linspace")
# y = tf.Variable([x])
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(y.eval())




# performace the algorithm of add
'''
state = tf.Variable(0)
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
'''

