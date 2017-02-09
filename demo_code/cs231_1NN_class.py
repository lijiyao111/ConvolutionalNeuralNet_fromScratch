import numpy as np 
import matplotlib.pyplot as plt 



def sigmoid(x):
  return 1/(1+np.exp(-x))

def deriv_sig(x):
  return sigmoid(x)*(1-sigmoid(x))

def relu(x):
  return np.maximum(0,x)

def deriv_relu(x):
  return 1.*(x>0)

def tanh(x):
  return np.tanh(x)

def deriv_tanh(x):
  return 1. - np.power(tanh(x),2) 

class OneLayerNN():

  def __init__(self, input_dim=2, hidden_dim=100,num_classes=10,activation='relu',reg=0.0):
    h1=input_dim
    h2 = hidden_dim # size of hidden layer

    N1=np.sqrt(h1)
    self.W1 = np.random.randn(h1,h2)/N1
    self.b1 = np.zeros((1,h2))

    N2=np.sqrt(h2)
    self.W2 = np.random.randn(h2,num_classes)/N2
    self.b2 = np.zeros((1,num_classes))

    self.reg=reg
    self.activation=activation

  def predict(self,X):
    # evaluate training set accuracy
    a1=X
    z2=np.dot(a1, self.W1) + self.b1
    if self.activation=='sigmoid':
      a2=sigmoid(z2)
    elif self.activation=='tanh':
      a2=tanh(z2)
    else: ## by default, use ReLU activation
      a2=relu(z2)

    scores = np.dot(a2,self.W2) +self.b2
    predicted_class = np.argmax(scores, axis=1)
    # print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
    return predicted_class

  def train(self,X,y,Niter=1000,learning_rate=1e-1,Nprint=100,
    batch_size=100,lr_decay=1.0):
    num_train = X.shape[0]

    if batch_size==0:
      batch_size=num_train

    for i in range(Niter):

      batch_mask = np.random.choice(num_train, batch_size)
      X_batch = X[batch_mask]
      y_batch = y[batch_mask]

      a1=X_batch

      # evaluate class scores, [N x K]
      z2=np.dot(a1, self.W1) + self.b1
      if self.activation=='sigmoid':
        a2=sigmoid(z2)
      elif self.activation=='tanh':
        a2=tanh(z2)
      else: ## by default, use ReLU activation
        a2=relu(z2)

      z3=np.dot(a2,self.W2) +self.b2

      # compute the class probabilities
      exp_scores = np.exp(z3)
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
      
      # compute the loss: average cross-entropy loss and regularization
      corect_logprobs = -np.log(probs[range(batch_size),y_batch])
      data_loss = np.sum(corect_logprobs)/batch_size
      reg_loss = 0.5*self.reg*np.sum(self.W1*self.W1) 
      reg_loss += 0.5*self.reg*np.sum(self.W2*self.W2) 
      
      loss = data_loss + reg_loss
      if i % Nprint == 0:
        print("iteration %d: loss %f" % (i, loss))
      
      # compute the gradient on scores
      delta_d3 = probs
      delta_d3[range(batch_size),y_batch] -= 1
      delta_d3 /= batch_size
      
      # backpropate the gradient to the parameters
      # first backprop into parameters W3 and b3

      dW2 = np.dot(a2.T, delta_d3)
      db2 = np.sum(delta_d3, axis=0, keepdims=True)

      if self.activation=='sigmoid':
        delta_d2 = np.dot(delta_d3, self.W2.T)*deriv_sig(z2)
      elif self.activation=='tanh':
        delta_d2 = np.dot(delta_d3, self.W2.T)*deriv_tanh(z2)
      else: ## by default, use ReLU activation
        delta_d2 = np.dot(delta_d3, self.W2.T)*deriv_relu(z2)

      # finally into W,b
      dW1 = np.dot(a1.T, delta_d2)
      db1 = np.sum(delta_d2, axis=0, keepdims=True)
      
      # add regularization gradient contribution
      dW2 += self.reg * self.W2
      dW1 += self.reg * self.W1
      
      # perform a parameter update
      learning_rate *= lr_decay
      self.W1 += -learning_rate * dW1
      self.b1 += -learning_rate * db1
      self.W2 += -learning_rate * dW2
      self.b2 += -learning_rate * db2

if __name__=='__main__':

  N = 100 # number of points per class
  D = 2 # dimensionality
  K = 3 # number of classes
  X = np.zeros((N*K,D)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels
  for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
  # lets visualize the data:
  # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  # plt.show()

  model=OneLayerNN(D,100,K,activation='relu',reg=1e-3)

  model.train(X,y,Niter=1000,learning_rate=1e-0,batch_size=N*K)


  y_pred=model.predict(X)
  print('training accuracy: %.2f' % (np.mean(y_pred == y)))

  # plot the resulting classifier
  h = 0.02
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
  Z=model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  fig = plt.figure()
  plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.show()