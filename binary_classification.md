# Here I will explain how i am training my neuron to predict binary classification

So what is binary classification? 
earlier in 01_ single neuron.py we were doing linear regression\
so we are predicitng a continuous value, like price of house or salary of person\
its prediction using numbers, plotting it, doing some cool maths and getting the line we need.

But what if we want to predict something that can only be 0 or 1? 

like if a person is going to buy a house or not 

so we use binary classification 

# The Problem
```
>We need to add one thing: an activation function called Sigmoid.

Why?

Linear neuron outputs: any number (-∞ to +∞)
But we want probability: a number between 0 and 1
Sigmoid squashes any number into the range [0, 1]
```

# Sigmoid function

σ(z) = 1 / (1 + e^(-z))

Examples:
σ(-10) = 0.000045  (very close to 0)
σ(0)   = 0.5       (right in the middle)
σ(10)  = 0.999955  (very close to 1)