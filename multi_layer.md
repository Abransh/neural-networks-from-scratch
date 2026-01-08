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

```
## Step 2: Hidden Layer → Output Layer
```
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
```

# What Did the Hidden Neurons Learn?

## This is a pretty thing, you must look at this 

You'll see something like:
```
Input    | Hidden1 | Hidden2 | Output | Expected
-------------------------------------------------------
[0 0]    |  0.076  |  0.012  |  0.003 |    0
[0 1]    |  0.942  |  0.031  |  0.998 |    1
[1 0]    |  0.951  |  0.029  |  0.998 |    1
[1 1]    |  0.993  |  0.986  |  0.002 |    0
``` 

Notice the pattern:

- Hidden1 (OR gate): High when ANY input is 1
  * [0,0] → 0.076 (low)
  * [0,1] → 0.942 (high!)
  * [1,0] → 0.951 (high!)
  * [1,1] → 0.993 (high!)

- Hidden2 (AND gate): High only when BOTH inputs are 1
  * [0,0] → 0.012 (low)
  * [0,1] → 0.031 (low)
  * [1,0] → 0.029 (low)
  * [1,1] → 0.986 (high!)


# Visual Decision Boundary

So if you are bored of maths and python
add this code into your file and run it (its already in my file as part3 )

```python
# Visualize decision boundary
x1_range = np.linspace(-0.5, 1.5, 300)
x2_range = np.linspace(-0.5, 1.5, 300)
xx1, xx2 = np.meshgrid(x1_range, x2_range)

# Compute predictions for entire grid
grid_points = np.c_[xx1.ravel(), xx2.ravel()]
z1_grid = grid_points @ W1 + b1
a1_grid = sigmoid(z1_grid)
z2_grid = a1_grid @ W2 + b2
a2_grid = sigmoid(z2_grid)
a2_grid = a2_grid.reshape(xx1.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx1, xx2, a2_grid, levels=20, cmap='RdYlBu_r', alpha=0.7)
plt.colorbar(label='Prediction (0=blue, 1=red)')
plt.contour(xx1, xx2, a2_grid, levels=[0.5], colors='black', linewidths=3)

# Plot data points
colors = ['blue' if label == 0 else 'red' for label in y.flatten()]
for i in range(4):
    plt.scatter(X[i, 0], X[i, 1], c=colors[i], s=300, 
               edgecolors='black', linewidth=3, zorder=5)
    plt.text(X[i, 0] + 0.08, X[i, 1] + 0.08, 
            f"{y[i,0]}", fontsize=14, fontweight='bold')

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('Input 1', fontsize=12)
plt.ylabel('Input 2', fontsize=12)
plt.title('XOR Decision Boundary (Non-Linear!)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()

```


![results](assets/image.png)


# so it concludes with with a fact that Backpropagation is the Magic of Training Deep Networks

Now I talk about some maths to understand backpropagation

So if you are a fan of maths and you like calculus you must know chain rule 
it says 
```
If y = f(g(x)), then:
dy/dx = (dy/dg) × (dg/dx)
```

Our network:
```
Input (x) → Hidden (h) → Output (y) → Loss (L)
```
**To update hidden layer weights, we need: dL/dW1**

Chain rule says:
```
dL/dW1 = dL/dy × dy/dh × dh/dW1
         └──┬──┘ └──┬──┘ └──┬──┘
            │       │       │
            │       │       │
         Output    Hidden   Local
         gradient  gradient gradient
```

***This is backpropagation - computing gradients layer by layer, backwards!***

So the file has new code to it 
as Backward Pass

written as 
```python
# ===== BACKWARD PASS (Backpropagation) =====
# Output layer gradients
dz2 = a2 - y                    # (4, 1)
dW2 = (a1.T @ dz2) / len(X)     # (2, 4) @ (4, 1) = (2, 1)
db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)

# Hidden layer gradients (backprop through sigmoid)
dz1 = (dz2 @ W2.T) * a1 * (1 - a1)  # (4, 1) @ (1, 2) * (4, 2) = (4, 2)
dW1 = (X.T @ dz1) / len(X)           # (2, 4) @ (4, 2) = (2, 2)
db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)
```

## what it means?
```
dz2 = a2 - y
```
**what is this?**
For Binary Cross-Entropy loss with sigmoid, the math works out beautifully:
```
dz2 = a2 - y
```


This is the derivative of loss w.r.t. the output layer's pre-activation (z2).

**Intuition:**
- If predicted 0.9 but true is 1 → error = -0.1 (should increase)
- If predicted 0.1 but true is 0 → error = +0.1 (should decrease)

**For our XOR example, one iteration:**
```
Input [1, 0], true output: 1
Predicted: 0.5 (initially)
dz2 = 0.5 - 1.0 = -0.5
```
This means: "output is too low by 0.5, need to increase it!"

## Step 2: Gradient w.r.t. Output Layer Weights (W2)

```
dW2 = (a1.T @ dz2) / len(X)
```
**Chain rule:** 
```
dL/dW2 = dL/dz2 × dz2/dW2
```
Recall the forward pass:
```
z2 = a1 @ W2 + b2
```
**Derivative of z2 w.r.t. W2:**
```
dz2/dW2 = a1.T  (this is why we use a1.T in the gradient!)
```

## Full gradient:

```python
dW2 = a1.T @ dz2
     └──┬──┘ └─┬─┘
        │      │
   Activations Error
   from hidden from output
```

- Intuition: "How much did each hidden neuron contribute to the error?"

  * If hidden neuron had high activation AND error is high → big gradient
  * If hidden neuron was off (low activation) → small gradient (it wasn't involved)


## Step 4: Gradient w.r.t. Hidden Layer Weights (W1)
  


Still there are challenges to it 

> The challenge:The hidden layer doesn't see the actual error directly - it's hidden! We need to propagate the error backwards.