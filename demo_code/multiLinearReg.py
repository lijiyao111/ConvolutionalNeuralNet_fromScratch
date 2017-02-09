import numpy as np 
import matplotlib.pyplot as plt 

Nd=1000
Nx=10

X=np.zeros((Nd,Nx))
in_x=(np.random.random(Nd)-0.0)*1
in_x=np.sort(in_x)

# plt.plot(in_x,'.')
# plt.show()

for i in range(Nx):
    X[:,i]=in_x**(i+1)

w=np.array([1,2,1,0,0,0,0,0,0,0])
b=5

y=np.dot(X,w)+b+np.random.randn(Nd)*0.2

w_pred=np.zeros(Nx)
b_pred=0


nloop=500
step=1e-1
reg=1e-3


for l in range(nloop):

    delta_w=np.zeros(Nx)
    delta_b=0
    for i in range(Nd):
        # for j in range(Nx):
        delta_w += (y[i]-np.dot(X[i,:],w_pred)-b_pred)*(-X[i,:])
        delta_b += (y[i]-np.dot(X[i,:],w_pred)-b_pred)*(-1)
    w_pred += -step*((1/Nd)*delta_w+reg*w_pred)
    b_pred += -step*(1/Nd)*delta_b

print('Training is Done!')

y_pred=np.dot(X,w_pred)+b_pred

# y_pred=k*x+h
plt.plot(in_x,y,'.')
plt.plot(in_x,y_pred,'r')
plt.show()