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



# Whats different in 02_binary_classification.py 

1. earlier we did regression 
```python
X = np.random.rand(100, 1) * 10  # (100, 1) - ONE feature
y = 2 * X + 1 + noise             # Continuous numbers
```

2. This time (classification):
```python 
X = np.vstack([X_class0, X_class1])  # (100, 2) - TWO features!
y = [0, 0, 0, ..., 1, 1, 1]          # Binary labels: 0 or 1
```

**Why 2 features?**
- Feature 1: x-coordinate
- Feature 2: y-coordinate
- This lets us visualize the decision boundary as a line in 2D!

### **2. The Sigmoid Function**

Look at the table you printed:
```
Input  | Output
--------------------
-10.0  | 0.000045  ← very confident it's class 0
 -5.0  | 0.006693
 -1.0  | 0.268941
  0.0  | 0.500000  ← uncertain (could be either)
  1.0  | 0.731059
  5.0  | 0.993307
 10.0  | 0.999955  ← very confident it's class 1
```
3. The New Loss Function: Binary Cross-Entropy

Old loss (MSE for regression):

loss = np.mean((y_pred - y) ** 2)

New loss (BCE for classification):
loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
```

**Why different?**

MSE doesn't work well for probabilities. Let me show you why:

**Example with MSE (bad for classification):**
```
True label: y = 1
Predicted probability: 0.9

MSE loss = (0.9 - 1)² = 0.01  (seems small, okay)

But if predicted: 0.1
MSE loss = (0.1 - 1)² = 0.81  (bigger, but not by much)
```

**Same example with Cross-Entropy (good!):**
```
True label: y = 1
Predicted: 0.9
BCE = -log(0.9) = 0.105

Predicted: 0.1
BCE = -log(0.1) = 2.303  (MUCH WORSE! 20x larger!)

**Cross-Entropy heavily penalizes confident wrong predictions!**

How BCE Works:

loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
         ───────┬──────────────      ─────────┬────────────────────
                │                             │
         When y=1 (true class 1)         When y=0 (true class 0)

