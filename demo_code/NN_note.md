# Loss function and Gradient
In neural network, there are lots of linear relationships between the neurons (variables within the neural network). 

Take the simpliest case, where the input $X$ is of dimension $N\times D$, with $D$ features and $N$ instances. There are $C$ classes to be identified. ($x_i$, $y_i$) is one instance, where $x_i$ is a row of $X$ and $y_i$ is a number to represent one class. $y_i \in \\{0 \dots C-1 \\}$ The weight parameter of the linear relationship is $W$, of dimension $D \times C$. There is also another parameter called bias, $b$, which just give some shift and has dimension of $C$.

To simplify the notations, we define,
$f(x_i;W,b)=W^T x_i+b$
In which,
$w_j$ is the $j$th column of $W$, $b_j$ is the $j$th item of $b$.

We also define,
$f_j^i=f(x_i;W,b)_j=w_j^T x_i+b_j$


# Loss function
The total loss functions is defined to be a combination of data loss and regularization term on parameters.
$$
L=\frac{1}{N}\sum_i^{N} L_i+\frac{1}{2}\lambda||W||^2
$$

Here the regularization on weight is defined as L2 norm.

# Gradient

$$
\frac{\partial L}{\partial w_j}=\frac{1}{N}\sum_i^{N}\frac{\partial L_i}{\partial w_j}+\lambda w_j \\\
\frac{\partial L}{\partial b_j}=\frac{1}{N}\sum_i^{N}\frac{\partial L_i}{\partial b_j}
$$

Both the loss function and the form of gradient calculation are in general form. Different regression methods defined different form of $L_i$.


# Binary classes
For binary classes, we just need to predict one of the two classes. If one class is less likely, then the other class should be the most possible one. 

In binary classes case, the weights collapse from matrix $W$ to a vector $w$ of size $C$. The bias also collapse from a vector to a scalar, but still denoted as $b$ for reason of being lazy. 

## Logistic regression
In logistic regression, define the probability of $y$ as $p(y)=\sigma(y)$,
where $\sigma(y)$ is the "sigmoid" or "logistic" function,
$\sigma(y)=\frac{1}{1+e^{-y}}$. $y_i \in \\{0,1\\}$. To predict $y$ with $x_i$, simply check wether $p(f(x_i;w,b))$ is closer to 1 or 0. (Or just check the sign of $f(x_i;w,b)$). Logistic regression nicely give the probability of the prediction in addition to the prediction itself. 

**Loss**
$$
L_i=-(y_i \log(p(f(x_i;w,b)))+(1-y_i)\log(1-p(f(x_i;w,b))))
$$

**Gradient**
$$
\frac{\partial L}{\partial w}=x_i(p(f(x_i;w,b))-y_i) \\\
\frac{\partial L}{\partial b}=p(f(x_i;w,b))-y_i
$$


## Binary SVM
In binary SVM, for convenience purpose, set $y_i \in \\{-1,1\\}$, To predict $y$ with $x_i$, simply check wether $f(x_i;w,b)$ is positive (indicate $y_i = 1$) or negative (indicate $y_i = -1$). 

**Loss**
$$
L_i=C\max(0,1 -  y_i f(x_i;w,b) )
$$

$C$ is a hyper-paramter. However, because we have included the regularization, which has the same effect, $C$ can be set to one. 

**Gradient**
$$
\frac{\partial L}{\partial w} =
\begin{cases}
- x_i y_i,  & \text{if $1 -  y_i f(x_i;w,b)>0$} \\\
0, & \text{else}
\end{cases}
$$

$$
\frac{\partial L}{\partial b} =
\begin{cases}
- y_i,  & \text{if $1 -  y_i f(x_i;w,b)>0$} \\\
0, & \text{else}
\end{cases}
$$



# Multiple classes

## Softmax regression
Softmax regression is a general form of logistic regression for multiclass case. In softmax regression, define probability $p(y=k|x_i)=\frac{e^{f_{k}^i}}{\sum_j e^{f_j^i}}$. To predict, just check which $k$ gives the largest probability $p(y=k|x_i)$.

**Loss**
$$
L_i=-\log(p(y_i|x_i))
$$

**Gradient**
$$
\frac{\partial L_i}{\partial w_j}=x_i(p(y_i|x_i)-1(y_i=j)) \\\
\frac{\partial L_i}{\partial b_j}=p(y_i|x_i)-1(y_i=j)
$$


## Multiclass SVM
Multiclass SVM regression is a general form of binary SVM regression for multiclass case. To predict, just check which $k$ gives the largest value of $f(x_i;W,b)_k$.

**Loss**
$$
L_i=\sum_{j \neq y_i} \max(0,f_j^i -f_{y_i}^i + \Delta )
$$

$\Delta$ is the margin, a hyper-paramter. However, because we have included the regularization, which has the same effect, $\Delta$ can be set to one. 

**Gradient**
for $j=y_i$
$$
\frac{\partial L_i}{\partial w_{y_i}}=-(\sum_{j\neq y_i} 1(f_j^i - f_{y_i}^i+\Delta>0))x_i \\\
\frac{\partial L_i}{\partial b_{y_i}}=-\sum_{j\neq y_i} 1(f_j^i - f_{y_i}^i+\Delta>0)
$$

for $j\neq y_i$
$$
\frac{\partial L_i}{\partial w_j}=1(f_j^i - f_{y_i}^i+\Delta>0)x_i \\\
\frac{\partial L_i}{\partial b_j}=1(f_j^i - f_{y_i}^i+\Delta>0)
$$




