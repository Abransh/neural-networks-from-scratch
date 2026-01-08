# multi_layer.md

## So what we did now to solve the XOR problem

We have been talking about this a lot, because its simply very interesting to see how a multi layer network can solve the XOR problem. 

What is Multi Layer Network?
What is multiple neurons in a layer?

We will talk about a lot in this file. 

# What We Built - The Architecture
```
Input Layer    Hidden Layer      Output Layer
(2 neurons)    (2 neurons)       (1 neuron)

   x1  ─┐
        ├─→ [h1] ─┐
   x2  ─┤         ├─→ [output]
        └─→ [h2] ─┘

Total: 9 parameters
- W1: (2×2) = 4 weights
- b1: (1×2) = 2 biases
- W2: (2×1) = 2 weights
- b2: (1×1) = 1 bias
```

First we do  Forward Pass - Step by Step

## Step 1: Input Layer → Hidden Layer
```
z1 = X @ W1 + b1
a1 = sigmoid(z1)
```

**What happens:**
```
Input: [1, 0]

z1[0] = w1[0,0]*1 + w1[1,0]*0 + b1[0]  # First hidden neuron
z1[1] = w1[0,1]*1 + w1[1,1]*0 + b1[1]  # Second hidden neuron

Then apply sigmoid to get activations a1
```

**After training, the learned weights might be:**
```
W1 = [[ 5.2,  4.8],   # Weights from x1 to [h1, h2]
      [ 5.1,  4.9]]   # Weights from x2 to [h1, h2]

b1 = [[-2.5, -7.3]]   # Biases for [h1, h2]
```

**For input [1, 0]:**
```
z1[0] = 5.2*1 + 5.1*0 - 2.5 = 2.7  →  sigmoid(2.7) = 0.94
z1[1] = 4.8*1 + 4.9*0 - 7.3 = -2.5 →  sigmoid(-2.5) = 0.08

Hidden layer outputs: [0.94, 0.08]


- Step 2: Hidden Layer → Output Layer

z2 = a1 @ W2 + b2
a2 = sigmoid(z2)
```

**After training, weights might be:**
```
W2 = [[ 7.2],    # Weight from h1 to output
      [-7.1]]    # Weight from h2 to output

b2 = [[-3.2]]    # Bias for output
```

**For hidden activations [0.94, 0.08]:**
```
z2 = 0.94*7.2 + 0.08*(-7.1) - 3.2 = 3.2
output = sigmoid(3.2) = 0.96 ≈ 1 ✓

# What Did the Hidden Neurons Learn?