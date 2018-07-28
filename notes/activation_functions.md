# Notes on basic activation functions used in Deep Learning

Activation functions are used in Neural Networks to decide if a neuron should be activated or not.
The output from the layer is fed through the activation function and hence transform it to see if it should be activated or not.
During backpropogation the derivative of the activation function is calculated to find a local minima for the gradient and reduce the loss value.

## Activation Functions
* Binary Step
* Linear
* Sigmoid
* Tanh
* ReLU
* Leaky ReLU
* Softmax

### Binary Step
```
f(x) = 1, x >= 0
     = 0, x < 0
f'(x) = 0, for all x
```
Binary Step function is just like the name suggests. It either activates the function or it doesnt. This is a good choice when the classification is only between something like a Yes or a No. But usually we want more than 2 classifications. Also since its either 0 or 1 the derivative of this function will be `0` making it `incapable of learning`.

### Linear
```
f(x) = ax
f'(x) = a
```
The activation is directly proportional to the input. But since the derivative of a linear function is a constant which means it is independent of the input x. This means that everytime during backpropogation the gradient will remain the same. So there will be no improvement in the error since the gradient remains the same everytime.

### Sigmoid
```
f(x) = 1/(1 + e^-x)
f'(x) = f(x)(1-f(x))
```
Non linear function. Derivative is also non linear. Range: [0,1]. Used when we need to predict the probability as an output. Derivative ranges from [0,0.25]. Used for binary classification in logistic regression.


### Tanh
```
f(x) = tanh(x) = 2*sigmoid(2x) - 1
f'(x) = 1 - f(x)^2
```
### ReLU
```
f(x)=max(0,x)
f'(x) = 1, x>=0 
      = 0, x<0
```

### Leaky ReLU
```
f(x) = ax, x<0
     = x, x>=0
```

### Softmax
```
f(z)j = e^zj/(sum(k=1->K)(e^zk))  (for j = 1 .. K)
```
