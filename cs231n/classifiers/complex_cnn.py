import numpy as np
from pprint import pprint

from cs231n.layers import *
from cs231n.layer_utils import *


class MineConvNet(object):
  '''
  (conv-[spatial batch norm]-relu-pool) X N - (affine-[batch norm]-relu-[dropout]) X M - affine - softmax 
  '''

  def __init__(self, input_dim=(3,32,32), num_filters=[10,10], filter_size=7, 
              hidden_dims=[10,10], num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0, 
               weight_scale=1e-3, use_autoweight=False, dtype=np.float32, seed=None):

    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_hidLayer = len(hidden_dims)
    self.num_filLayer = len(num_filters)
    self.dtype = dtype
    self.params = {}

    if seed is not None:
      np.random.seed(seed)

    if type(num_filters) != list:
      raise ValueError('num_filters has to be a list')

    if type(hidden_dims) != list:
      raise ValueError('hidden_dims has to be a list')

    C, H, W = input_dim
    stride_conv=1

    ## Conv Layer
    for i in range(self.num_filLayer):
      if i==0:
        # C=C
        F=num_filters[i]
        HH=filter_size
        WW=filter_size
      else:
        C=num_filters[i-1]
        F=num_filters[i]
        HH=filter_size
        WW=filter_size

      W_conv_dim=(F,C,HH, WW)
      if use_autoweight:
        N1=C*HH*WW
        self.params['WC'+str(i+1)]=np.random.randn(*W_conv_dim)*np.sqrt(2.0/N1)
      else:
        self.params['WC'+str(i+1)]=np.random.randn(*W_conv_dim)*weight_scale
      self.params['bC'+str(i+1)]=np.zeros(F)

      if self.use_batchnorm:
        self.params['gammaC'+str(i+1)]=np.ones(F)
        self.params['betaC'+str(i+1)]=np.zeros(F)


    Hout,Wout=self.size_Conv(stride_conv, filter_size, H, W, self.num_filLayer)
    # print H,W,Hout,Wout,F

    # print 'Here'
    # print self.size_Conv(stride_conv, filter_size, H, W, 1)
    # print self.size_Conv(stride_conv, filter_size, H, W, 2)


    ## affine layer
    for i in range(self.num_hidLayer):
      if i==0:
        Nin=Hout*Wout*F
        Nout=hidden_dims[i]
      else:
        Nin=hidden_dims[i-1]
        Nout=hidden_dims[i]

      W_aff_dim=(Nin,Nout)
      # print W_aff_dim
      if use_autoweight:
        N1=Nin
        self.params['WF'+str(i+1)]=np.random.randn(*W_aff_dim)*np.sqrt(2.0/N1)
      else:
        self.params['WF'+str(i+1)]=np.random.randn(*W_aff_dim)*weight_scale
      self.params['bF'+str(i+1)]=np.zeros(Nout)

      if self.use_batchnorm:
        self.params['gammaF'+str(i+1)]=np.ones(Nout)
        self.params['betaF'+str(i+1)]=np.zeros(Nout)

    ## Last layer before softmax
    if use_autoweight:
      N1=hidden_dims[-1]
      self.params['WSc']=np.random.randn(hidden_dims[-1],num_classes)*np.sqrt(2.0/N1)
    else:
      self.params['WSc']=np.random.randn(hidden_dims[-1],num_classes)*weight_scale
    self.params['bSc']=np.zeros(num_classes)


    # print 'Initialization Done!'
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_filLayer +self.num_hidLayer)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


    ## calculate the stupid conv net dimention, why not providing parameters?
  def size_Conv(self, stride_conv, filter_size, H, W, Nbconv):
      P = (filter_size - 1) / 2  # padd
      Hc = (H + 2 * P - filter_size) / stride_conv + 1
      Wc = (W + 2 * P - filter_size) / stride_conv + 1
      width_pool = 2
      height_pool = 2
      stride_pool = 2
      Hp = (Hc - height_pool) / stride_pool + 1
      Wp = (Wc - width_pool) / stride_pool + 1
      if Nbconv == 1:
          return Hp, Wp
      else:
          H = Hp
          W = Wp
          return self.size_Conv(stride_conv, filter_size, H, W, Nbconv - 1)



  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input: X (N, C, H, W)
    Output: y (N,) 
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for i in range(len(self.bn_params)):
        self.bn_params[i]['mode']=mode
      # for bn_param in self.bn_params: ## list of dict
      #   bn_param['mode'] = mode


    scores=None


    #############################################################
    # Forward propagation
    #############################################################
    # pass conv_param to the forward pass for the convolutional layer

    convL={}

    bnInd=0

    for i in range(self.num_filLayer):

      W=self.params['WC'+str(i+1)]
      b=self.params['bC'+str(i+1)]

      filter_size = W.shape[2]
      conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

      # pass pool_param to the forward pass for the max-pooling layer
      pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

      bn_param=self.bn_params[bnInd]
      bnInd += 1

      if i!=0:
        X=convL['c'+str(i)]
      # else:
      #   X=X
      if self.use_batchnorm:
        gammaC=self.params['gammaC'+str(i+1)]
        betaC=self.params['betaC'+str(i+1)]
        # print gammaC.shape,betaC.shape,X.shape,W.shape,b.shape
        conv_out,conv_cache=conv_norm_relu_pool_forward(X,W,b,
            gammaC, betaC,conv_param, pool_param, bn_param)
        # print conv_out.shape
      else:
        conv_out,conv_cache=conv_relu_pool_forward(X,W,b,conv_param,pool_param)
      # print 'conv in', X.shape
      # print 'conv out',conv_out.shape
      convL['c'+str(i+1)]=conv_out
      convL['c_cache'+str(i+1)]=conv_cache


    ## affine layers
    affL={}
    for i in range(self.num_hidLayer):
      if i==0:
        X=convL['c'+str(self.num_filLayer)]
      else:
        X=affL['h'+str(i)]

      if self.use_dropout:
        X,affL['hdrop_cache'+str(i+1)]=dropout_forward(X,self.dropout_param)

      bn_param=self.bn_params[bnInd]
      bnInd += 1

      W=self.params['WF'+str(i+1)]
      b=self.params['bF'+str(i+1)]
      # print X.shape
      # print W.shape
      # print b.shape
      if self.use_batchnorm:
        gammaF=self.params['gammaF'+str(i+1)]
        betaF=self.params['betaF'+str(i+1)]
        aff_layer,aff_cache=affine_norm_relu_forward(X,W,b,
          gammaF,betaF,bn_param)
      else:
        aff_layer,aff_cache=affine_relu_forward(X,W,b)
      # print 'W'
      # pprint(W)
      # print 'b' 
      # pprint(b)
      # print 'X'
      # pprint(X)
      # print 'out'
      # pprint(aff_layer)
      affL['h'+str(i+1)]=aff_layer
      affL['h_cache'+str(i+1)]=aff_cache

    ## Score layer before softmax
    hidden_layer=affL['h'+str(self.num_hidLayer)]
    if self.use_dropout:
      hidden_layer,hdrop_cache=dropout_forward(hidden_layer,self.dropout_param)

    W=self.params['WSc']
    b=self.params['bSc']


    # print 'Ws'
    # pprint(W)
    # print 'bs'
    # pprint(b)
    # print 'Xs'
    # pprint(hidden_layer)
    scores,scores_cache=affine_forward(hidden_layer,W,b)
    # print 'scores'
    # pprint(scores)
    # pprint(np.dot(hidden_layer,W)+b)


    ## if test mode return early
    if mode=='test':
      return scores


    loss, grads = 0.0, {} 
    #############################################################
    # Backward propagation
    #############################################################
    data_loss, dscores = softmax_loss(scores,y)
    reg_loss=0.0
    for i in range(self.num_filLayer):
      reg_loss += 0.5*self.reg*np.sum((self.params['WC'+str(i+1)])**2)
    for i in range(self.num_hidLayer):
      reg_loss += 0.5*self.reg*np.sum((self.params['WF'+str(i+1)])**2)
    reg_loss += 0.5*self.reg*np.sum((self.params['WSc'])**2)

    loss=data_loss+reg_loss

    ## Score layer before softmax
    dhidden, dWSc, dbSc =affine_backward(dscores,scores_cache)
    # print 'dW0'
    # pprint(dWSc)
    dWSc += self.reg*self.params['WSc']
    grads['WSc'] = dWSc
    grads['bSc'] = dbSc
    # print 'dW'
    # pprint(dWSc)
    # print 'db'
    # pprint(dbSc)
    # print 'dX'
    # pprint(dhidden)

    if self.use_dropout:
      dhidden=dropout_backward(dhidden,hdrop_cache)

    ## affine layers
    for i in range(self.num_hidLayer-1,-1,-1):
      # print 'daffine',i
      if i==self.num_hidLayer-1:
        daff_old=dhidden
      else:
        daff_old=affL['dh'+str(i+1+1)] # come from the layer next
      aff_cache=affL['h_cache'+str(i+1)]

      if self.use_batchnorm:
        daff,dWF,dbF, dgammaF, dbetaF=affine_norm_relu_backward(daff_old,aff_cache)
        grads['gammaF'+str(i+1)]=dgammaF
        grads['betaF'+str(i+1)]=dbetaF
      else:
        daff,dWF,dbF=affine_relu_backward(daff_old,aff_cache)

      if self.use_dropout:
        daff=dropout_backward(daff,affL['hdrop_cache'+str(i+1)])
        
      # print 'dW0'
      # pprint(dWF)
      affL['dh'+str(i+1)]=daff
      dWF += self.reg*self.params['WF'+str(i+1)]
      grads['WF'+str(i+1)] = dWF
      grads['bF'+str(i+1)] = dbF
      # print 'dW'
      # pprint(dWF)
      # print 'db'
      # pprint(dbF)
      # print 'dX'
      # pprint(daff)


    ## Conv layers
    for i in range(self.num_filLayer-1,-1,-1):
      # print 'dconv',i
      if i==self.num_filLayer-1:
        dconv_old=affL['dh'+str(1)]
      else:
        dconv_old=convL['dc'+str(i+1+1)] # come from the layer next
      conv_cache=convL['c_cache'+str(i+1)]

      if self.use_batchnorm:
        dconv,dWC,dbC,dgammaC, dbetaC=conv_norm_relu_pool_backward(dconv_old,conv_cache)
        grads['gammaC'+str(i+1)]=dgammaC
        grads['betaC'+str(i+1)]=dbetaC
      else:
        dconv,dWC,dbC=conv_relu_pool_backward(dconv_old,conv_cache)
      convL['dc'+str(i+1)]=dconv
      dWC += self.reg*self.params['WC'+str(i+1)]
      grads['WC'+str(i+1)] = dWC
      grads['bC'+str(i+1)] = dbC


    return loss, grads


