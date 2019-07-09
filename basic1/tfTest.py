import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
a=3
# create a variable
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
y = tf.matmul(w,x)
print (w)
# variables have to be explicitly initialized before you can run Ops
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    #print(sess.run(w))
    #print(sess.run(y))
    #print(y.eval())
    print(w.eval())
    tf.print