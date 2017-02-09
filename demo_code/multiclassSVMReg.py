import matplotlib.pyplot as plt 
import numpy as np 

N = 100 # number of points per class
Nx = 2 # dimensionality
K = 3 # number of classes
Ny=K
X = np.zeros((N*K,Nx)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

Nd=N*K

# for j in range(K):
#   ix = range(N*j,N*(j+1))
#   r = np.linspace(0.0,1,N) # radius
#   t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
#   X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
#   y[ix] = j


X[0:N,:]=np.random.randn(N,2)*0.1+0.3
X[N:2*N,:]=np.random.randn(N,2)*0.1-0.3
X[2*N:3*N,0]=np.random.randn(N)*0.1-0.5
X[2*N:3*N,1]=np.random.randn(N)*0.1+0.5
y[0:N]=0
y[N:2*N]=1
y[2*N:3*N]=2


# # lets visualize the data:
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()


def cost(X,y):
    cost_sum =0
    for i in range(len(X)):
      delta = 1.0
      scores = np.dot(X[i,:],W_pred)+b_pred
      # compute the margins for all classes in one vector operation
      margins = np.maximum(0, scores - scores[y[i]] + delta)
      margins[y[i]] = 0
      cost_sum += np.sum(margins)
    cost_sum = 0.5*cost_sum/len(X)
    cost_sum += 0.5*reg*np.sum(W_pred*W_pred)
    return cost_sum

W_pred=np.zeros((Nx,Ny))
b_pred=np.zeros(Ny)



nloop=500
step=1e0
reg=1e-4


for l in range(nloop):

    delta_W=np.zeros((Nx,Ny))
    delta_b=np.zeros(Ny)
    for i in range(Nd): # m
        scores = X[i].dot(W_pred)+b_pred
        correct_class_score = scores[y[i]]

        for j in range(Ny):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                delta_W[:,y[i]] -= X[i,:]
                delta_b[y[i]] -= 1
                delta_W[:,j] += X[i,:]
                delta_b[j] += 1
    W_pred += -step*((1/Nd)*delta_W+reg*W_pred)
    b_pred += -step*(1/Nd)*delta_b

    if l%50 ==0:
        loss=cost(X,y)
        print('loop {} with loss {}'.format(l,loss))

print('Done')


y_pred=np.argmax(np.dot(X,W_pred)+b_pred,axis=1)
print('Accuracy: {}'.format(np.mean(y_pred==y)))


h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


Z=np.zeros(len(xx)*len(xx[0]))

for i in range(len(xx)*len(xx[0])):
    # print(softmax(np.dot(W_pred.T,np.c_[xx.ravel(),yy.ravel()][i,:])+b_pred))
    Z[i]=np.argmax(np.dot(W_pred.T,np.c_[xx.ravel(),yy.ravel()][i,:])+b_pred)

Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.scatter(X[:, 0], X[:, 1], c=oneOrzero(sigmoid(np.dot(X,w_pred)+b_pred)), s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

