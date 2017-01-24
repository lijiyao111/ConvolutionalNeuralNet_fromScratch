import numpy as np 
N,C,H,W=100,10,30,40
x=np.ones((N,C,H,W))

print x.shape

print np.var(x,axis=(0,2,3))#.shape

b= np.sum(x,axis=(0,2,3)).reshape(1,C,1,1)

print b.shape

print b.reshape(-1)

