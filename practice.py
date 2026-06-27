import numpy as np
from numpy import newaxis


rg = np.random.default_rng()

a = np.array([[0,1,2,3],
              [4,5,6,7],
              [8,9,10,11]])

# b = a
# print(b is a)

# def f(x):
#     print(id(x))

# print(id(a))
# f(a)

c=a.view()
# print(c)
# print(c is a)

# print(c.base is a) # c is a view of the data owned by a

# print(c.flags.owndata)

c = c.reshape((2,6)) # this will have no effect on a's shape
print(a.shape)