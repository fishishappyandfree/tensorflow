import tensorflow as tf

v1 = tf.Variable(tf.random_normal([1,2]), name='v1')
v2 = tf.Variable(tf.random_normal([2,3]), name='v2')
saver = tf.train.Saver()

with tf.Session() as sess:
    '''
    to save the all variable and operation
    '''
    # sess.run(tf.global_variables_initializer())
    # print('V1:', sess.run(v1))
    # print('v2', sess.run(v2))
    # save_path = tf.train.Saver().save(sess, 'save/model.ckpt')
    # print('model saved in file: ', save_path)

    '''
    to use the variable and operation
    '''
    saver.restore(sess, 'save/model.ckpt')
    print('v1', sess.run(v1))
    print('v2', sess.run(v2))
    print('model restored')
