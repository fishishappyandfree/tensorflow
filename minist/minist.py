import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import input_data

print("packs loaded")

print("Download and Extract MNIST dataset")
minist = input_data.read_data_sets('data/', one_hot=True)

# print("type of 'minist' is %s" % (type(minist)))
# print("number of train data is %d" % (minist.train.num_examples))
# print("number of test data is %d" % (minist.test.num_examples))

# What does the data of MINIST look like?
print("what does the data of MNIST look like?")
trainimg = minist.train.images
trainlabel = minist.train.labels
testimg = minist.test.images
testlabel = minist.test.labels

# print("type of 'trainimg' is %s" % (type(trainimg)))
# print("type of 'trainlabel' is %s" % (type(trainlabel)))
# print("type of 'testimg' is %s" % (type(testimg)))
# print("type of 'testlabel' is %s" % (type(testlabel)))
# print("shape of 'trainimg' is %s" % (trainimg.shape,))
# print("shape of 'trainlabel' is %s" % (trainlabel.shape,))
# print("shape of 'testimg' is %s" % (testimg.shape,))
# print("shape of 'testlabel' is %s" % (testlabel.shape,))

# how does the training data look like?
print("how does the training data look like?")
nsample = 5
randidx = np.random.randint(trainimg.shape[0],size = nsample)

# plt.figure(1)
# count = 1
for i in randidx:
    curr_img = np.reshape(trainimg[i,:],(28,28))  # 28 by 28 matrix
    curr_label = np.argmax(trainlabel[i,:])  # Label
    # plt.subplot(nsample, 1, count)
    plt.matshow(curr_img, cmap = plt.get_cmap("gray"))
    plt.title(""+str(i)+"th Trainimg Data"+"Label is"+str(curr_label))
    print(""+str(i)+"th Training Data"+"Label is"+str(curr_label))
    # count = count + 1
plt.show()



# Batch Learning?
print("Batch Learning?")
batch_size = 100
batch_xs, batch_ys = minist.train.next_batch(batch_size)
print("type of 'batch_xs' is %s" % (type(batch_xs)))
print("type of 'batch_ys' is %s" % (type(batch_ys)))
print("shape of 'batch_xs' is %s" % (batch_xs.shape,))
print("shape of 'batch_ys' is %s" % (batch_ys.shape,))
# print("shape of 'batch_ys' is %s" % (type(batch_ys.shape,)))