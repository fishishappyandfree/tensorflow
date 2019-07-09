import collections
import numpy as np

# collections.defaultdict
# collections.Counter().items()
# sorted(ss,lambda x:x[-1])
# dict.items()
list1 =[[1,2,3],[4,5,6]]
list2 = ['a', 'b', 'c']
# a = zip(list1,list2)
# print(a)
# print(type(a))
# for i in a:
#     print(i)
#     print(type(i))

b = max(map(len,list2))
# for i in b:
#     print(i)
#     print(type(i))
# print(b)
#
# dict.get()

# a1 = range(10)
# print(a1)
# print(a1[0:2])


a = np.vstack(np.array(list1))
b = np.array(list1)
print(a)
print(b)
print(np.vstack((a,b)))

x = np.load('C:/Users/wzs/Desktop/tensorflow/classification/data/valid_data/6k_gang_n2.npy')
print(x.shape)

listq.



