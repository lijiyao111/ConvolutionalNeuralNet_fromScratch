from scipy import signal
import numpy as np

x=np.ones((3,3))
y=np.ones((2,2))

z=signal.convolve2d(x,y,'full')
print(z.shape)