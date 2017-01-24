import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - (spatial batch norm) - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, 
               use_spatialbatchnorm=False, reg=0.0,
               dtype=np.float32, seed=None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_spatialbatchnorm = use_spatialbatchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    if seed is not None:
      np.random.seed(seed)
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    F=num_filters
    HH=filter_size
    WW=filter_size
    stride_conv=1
    Hp, Wp = self.size_Conv(stride_conv,filter_size,H,W)
    W_conv_dim=(F,C,HH, WW)
    W_h_affine_dim=(F*Hp*Wp,hidden_dim)
    W_out_affine_dim=(hidden_dim,num_classes)
    self.params['W1']=np.random.randn(*W_conv_dim)*weight_scale
    self.params['b1']=np.zeros(F)
    self.params['W2']=np.random.randn(*W_h_affine_dim)*weight_scale
    self.params['b2']=np.zeros(hidden_dim)
    self.params['W3']=np.random.randn(*W_out_affine_dim)*weight_scale
    self.params['b3']=np.zeros(num_classes)

    if self.use_spatialbatchnorm:
        self.params['gamma1']=np.ones(F)
        self.params['beta1']=np.zeros(F)


    # print 'Initialization Done!'
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

## calculate the stupid conv net dimention, why not providing parameters?
  def size_Conv(self,stride_conv, filter_size, H, W):
    P = (filter_size - 1) / 2  # padd
    Hc = (H + 2 * P - filter_size) / stride_conv + 1
    Wc = (W + 2 * P - filter_size) / stride_conv + 1
    width_pool = 2
    height_pool = 2
    stride_pool = 2
    Hp = (Hc - height_pool) / stride_pool + 1
    Wp = (Wc - width_pool) / stride_pool + 1    
    # return Hc, Wc, Hp, Wp
    return Hp, Wp
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    # print self.params['W1']
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    if self.use_spatialbatchnorm:
        gamma1=self.params['gamma1']
        beta1=self.params['beta1']

    # print W1.shape
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    bn_param={}
    if self.use_spatialbatchnorm:
       bn_param['mode'] = mode 

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # print X.shape


    if self.use_spatialbatchnorm:
        conv_out,conv_cache=conv_norm_relu_pool_forward(X,W1,b1, 
            gamma1, beta1,conv_param, pool_param, bn_param)
    else:
        conv_out,conv_cache=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    hidden_layer,hidden_cache=affine_relu_forward(conv_out,W2,b2)
    scores,scores_cache=affine_forward(hidden_layer,W3,b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    # If test mode return early
    if mode == 'test':
        return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    data_loss, dscores = softmax_loss(scores,y)
    reg_loss=0.5*self.reg*np.sum(W1*W1) +\
              0.5*self.reg*np.sum(W2*W2)+0.5*self.reg*np.sum(W3*W3)
    loss=data_loss+reg_loss

    dhidden, dW3,db3=affine_backward(dscores,scores_cache)
    dW3 += self.reg*W3

    dconv,dW2,db2=affine_relu_backward(dhidden,hidden_cache)
    dW2 += self.reg*W2

    if self.use_spatialbatchnorm:
        dx,dW1,db1,dgamma, dbeta=conv_norm_relu_pool_backward(dconv,conv_cache)
    else:
        dx,dW1,db1=conv_relu_pool_backward(dconv,conv_cache)
    dW1 += self.reg*W1



    grads['W1']=dW1
    grads['b1']=db1
    grads['W2']=dW2
    grads['b2']=db2
    grads['W3']=dW3
    grads['b3']=db3

    if self.use_spatialbatchnorm:
        grads['gamma1']=dgamma
        grads['beta1']=dbeta


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
