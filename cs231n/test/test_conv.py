class MyException(Exception):
    pass


def conv_forward(x, w, b, conv_param,method='native'):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N,C,H,W=x.shape
  F,C,HH,WW=w.shape
  S=conv_param['stride']
  P=conv_param['pad']

  x_pad=np.pad(x,((0,0),(0,0),(P,P),(P,P)),'constant')

  Hout = 1 + (H + 2 * P - HH) / S
  Wout = 1 + (W + 2 * P - WW) / S

  out=np.empty((N,F,Hout,Wout))

  if method=='native':
    for i in range(N):
      for j in range(F):
        for k in range(Hout):
          for l in range(Wout):
            out[i,j,k,l]=np.sum(x_pad[i,:,k*S:HH+k*S,l*S:WW+l*S]*w[j,:,:,:])+b[j]

  elif method=='conv':
  # Below is to demonstrate that CNN forward is indeed doing convolve2d (only if stride = 1)
    for i in range(N):
      for j in range(F):
        tmp=np.zeros((Hout,Wout))
        for k in range(C):
          tmp[:,:] +=signal.convolve2d(x_pad[i,k,:,:],w[j,k,::-1,::-1])#,'valid')
        out[i,j,:,:]=tmp
  else:
    raise MyException("Only native and conv methods are allowed.")


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward(dout, cache,method='native'):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache

  N,C,H,W=x.shape
  F,C,HH,WW=w.shape
  S=conv_param['stride']
  P=conv_param['pad']

  x_pad=np.pad(x,((0,0),(0,0),(P,P),(P,P)),'constant')

  _,_,Hout,Wout=dout.shape

  # out=np.empty((N,F,Hout,Wout))
  dx_pad=np.zeros(x_pad.shape)
  dw=np.empty(w.shape)
  db=np.empty(b.shape)

  if method=='native':
    for i in range(F):
      for j in range(C):
        for k in range(HH):
          for l in range(WW):
            dw[i,j,k,l]=np.sum(x_pad[:,j,k:k+S*Hout:S,l:l+S*Wout:S]*dout[:,i,:,:])
      db[i]=np.sum(dout[:,i,:,:])

    for i in range(N):
      for j in range(F):
        for k in range(Hout):
          for l in range(Wout):
            dx_pad[i,:,k*S:k*S+HH,l*S:l*S+WW] +=w[j,:,:,:]*dout[i,j,k,l]

  elif method=='conv':
    if S!=1:
      raise MyException("In conv method, stride S has to be 1!")
  # using convolve2d, (only if stride = 1)
    for j in range(F):
      for k in range(C):
        tmp=np.zeros((HH,WW))
        for i in range(N):
          tmp[:,:] +=signal.convolve2d(x_pad[i,k,:,:],dout[i,j,::-1,::-1],'valid')
        dw[j,k,:,:]=tmp
      db[j]=np.sum(dout[:,j,:,:])

    for i in range(N):
      for k in range(C):
        tmp=np.zeros((H+2,W+2))
        for j in range(F):
          tmp[:,:] +=signal.convolve2d(w[j,k,:,:],dout[i,j,:,:],'full')
        dx_pad[i,k,:,:]=tmp

  else:
    raise MyException("Only native and conv methods are allowed.")


  dx=dx_pad[:,:,P:H+P,P:W+P]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db