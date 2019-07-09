import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# tf.train.Saver
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
y = tf.matmul(w,x)
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    print(y.eval())
# Do some work with the model
# Save the variable to disk
    save_path = saver.save(sess,"C://Users//wzs//Desktop//tensorShow//savePath")
    print("Model saved in file:", save_path)