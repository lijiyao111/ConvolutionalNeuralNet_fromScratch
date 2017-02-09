# Convolutional Neural Network implementation
In CNN, the input image $X$ is of dimension $N\times C \times H \times W$, the filter $F$ is of dimension $F\times C \times HH \times WW$, the output $Y$ is of dimension $N\times F\times Hd \times Wd$. In this note, for simplicity reason, we look at one input example ($N=1$), assume one color layer of input ($C=1$), only one filter ($F=1$), thus all the input, filter, output are 2d matrices.


## Forward propogation
$X$ is of dimension $H \times W$, $F$ is of dimension $HH\times WW$, if the padding size is $P$, the stride is $S$, then the output $Y$ has dimension $Hd\times Wd$.
$$
Hd=1+(H+2P-HH)/S \\\
Wd=1+(W+2P-WW)/S
$$

$$
Y_{kl}=\sum_{i=0}^{HH-1} \sum_{j=0}^{WW-1} F_{ij} X_{i+kS, j+lS}
$$

Or we can write this equation in another form
$$
Y_{kl}=\sum_{m=kS}^{HH-1+kS} \sum_{n=lS}^{WW-1+lS} F_{m-kS,n-lS} X_{mn}
$$

If we make a very simple neural network, without hidden layer and activation function. The loss function is defined as
$$
L=\sum_k^{Hd-1} \sum_l^{Wd-1} f(Y_{kl})
$$
$f()$ is some defined function to calculate the total loss. 

Then the Gradient $L$ with respect to $X$ and $F$ are
$$
\begin{align}
\frac{\partial L}{\partial F_{ij}}&=\sum_k^{Hd-1} \sum_l^{Wd-1} f'(Y_{kl})\frac{\partial Y_{kl}}{\partial F_{ij}} \nonumber \\\
&=\sum_k^{Hd-1} \sum_l^{Wd-1} f'(Y_{kl})X_{i+kS,j+lS} \nonumber
\end{align}
$$

$$
\begin{align}
\frac{\partial L}{\partial X_{mn}}&=\sum_k^{Hd-1} \sum_l^{Wd-1} f'(Y_{kl})\frac{\partial Y_{kl}}{\partial X_{mn}} \nonumber \\\
&=\sum_k^{Hd-1} \sum_l^{Wd-1} f'(Y_{kl})F_{m-kS,n-lS} \nonumber
\end{align}
$$
in which, $f'(Y)$ is of dimension $Hd\times Wd$, same as $Y$.

## Where is convolution? Why $180^\circ$ rotation?

OK, for simplicity, let's assume there is no padding and stride is one ($P=0,S=1$).

Then we get $Hd=1+H-HH, Wd=1+W-WW$

$$
Y_{kl}=\sum_{i=0}^{HH-1} \sum_{j=0}^{WW-1} F_{ij} X_{i+k, j+l}
$$
This is actually the cross-correlation of $F$ and $X$, $F\star X$. Cross-correlation and convolution are quite similar to each other, in terms of calculation. We know that $\bar F \ast X = F \star X$, where $\bar F$ is the $180^\circ$ rotation of $F$. $180^\circ$ rotation is just another way to say 2D flip.

If we flip the Filter right-left and up-down, $\bar F_{HH-1-i,WW-1-j}=F_{ij}$, then after some rearrangement of the index, we get
$$
Y_{kl}=\sum_{i=0}^{HH-1} \sum_{j=0}^{WW-1} \bar F_{ij} X_{k-i+HH-1, l-j+WW-1}
$$
Which is exactly 2D convolution, $\bar F \ast X$.
And after careful check of the index of X, it never goes out of bound and exactly traverses from $0$ to $H-1$ for $k-i+HH-1$, $0$ to $W-1$ for $l-j+WW-1$, means that this convolution has mode of "Valid".

For gradient, we also have
$$
\frac{\partial L}{\partial F_{ij}}=\sum_k^{Hd-1} \sum_l^{Wd-1} f'(Y_{kl})X_{i+k,j+l} 
$$

$$
\frac{\partial L}{\partial X_{mn}}=\sum_k^{Hd-1} \sum_l^{Wd-1} f'(Y_{kl})F_{m-k,n-l}
$$

It is clear to see that $\frac{\partial L}{\partial F_{ij}} = \bar f'(Y) \ast X $ in "Valid" mode and $\frac{\partial L}{\partial X_{mn}} = f'(Y) \ast F $ in "Full" mode. $\bar f'(Y)$ is the $180^\circ$ rotation of $f(Y)$.

>Note that when $S>1$, it is no longer standard cross-correlation of convolution. But I guess it might be a convention to call this method Convolutional Neural Network. 

## A better way to calculate gradient
In calculating $\frac{\partial L}{\partial X_{mn}}$, altough the equation is clear, the index of $F$ will go out of bound. For the out of bound index of $F$, we need to set them to be zeros. This can be done by some ```if else``` check, or just pad lots of zeros to $F$ to form a new filter. And it will not be easy to figure out correctly with padding $P>0$ and strike $S>1$.

A trick to calculate $\frac{\partial L}{\partial X_{mn}}$ is that, instead of fixing the index of $X$, then figuring out the index of $f'(Y)$ , we fix the index of $f'(Y)$, then figure out the index of $X$. 

$$
\frac{\partial L}{\partial X_{i+kS,j+lS}}=\sum_k^{Hd-1} \sum_l^{Wd-1}f'(Y_{kl})F_{ij}
$$
The index $i,j$ will traverse the whole size of the filter $F$, and there is no need to concern about the index of $X$. This method is actually to "Think backward" of the forward propogation. This idea was inspired by discussion with Siyuan Wang. 