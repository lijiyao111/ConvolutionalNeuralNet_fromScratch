import numpy as np 

x_shape = (2, 2, 2, 2)
# x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
x=np.random.randn(*x_shape)

print x

# print x.max(axis=3).max(axis=2)


for i in range(x_shape[0]):
    for j in range(x_shape[1]):
        print np.unravel_index(x[i,j,:,:].argmax(),(x_shape[2],x_shape[3]))