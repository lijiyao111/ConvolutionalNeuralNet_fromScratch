import numpy as np

x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)

print(x)

P=1

# x_pad=np.lib.pad(x,((1,1),(1,1)),'constant')

x_pad = np.pad(x, ((0,0), (0,0), (P,P), (P,P)), 'constant')

print(x_pad)