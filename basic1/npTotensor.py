import tensorflow as tf

import numpy as np
a = np.zeros((3,3))
# print(a)
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(type(sess.run(ta)))
    print(a.ctypes)
