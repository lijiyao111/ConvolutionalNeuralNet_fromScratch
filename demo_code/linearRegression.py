import numpy as np 
import matplotlib.pyplot as plt 

Nd=1000
x=np.random.randn(Nd,1)
k_real=10
h_real=10
y=k_real*x+h_real+np.random.randn(Nd,1)*5

# plt.plot(x,y,'.')
# plt.show()

nloop=1000
step=1e-1

k=0
h=0
for l in range(nloop):

    delta_k=0
    delta_h=0
    for i in range(Nd):
        delta_k += (y[i]-k*x[i]-h)*(-x[i])
        delta_h += (y[i]-k*x[i]-h)*(-1)
    k += -step*(1/Nd)*delta_k
    h += -step*(1/Nd)*delta_h


y_pred=k*x+h
plt.plot(x,y,'.')
plt.plot(x,y_pred,'r')
plt.show()
