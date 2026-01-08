# This is continuation of xor problem the previous file 03_xor_problem.py
import numpy as np 
import matplotlib.pyplot as plt 

print("\n" + "="*60)
print("ATTEMPT 2: Neural Network with Hidden Layer (This WILL Work!)")
print("="*60)

# network architecture 
# input(2) -> hidden layer ( 2 neurons ) -> output (1)

np.seed.random(42)

X= np.array([[0,0], [0,1], [1,0], [1,1]])

y= np.array([[0],[1],[1],[0]])

def sigmoid(z): 
    return 1/ (1 + np.exp(-z))

# Layer 1: Input (2) → Hidden (2)

W1 = np.random.randn(2, 2) * 0.5
b1 = np.random.randn(1, 2) * 0.5  
# you can use b1 = np.zeroes((1,2)) as well 

# Layer 2: Hidden (2) → Output (1)

W2 = np.random.randn(2,1) * 0.5
b2 = np.zeros((1, 1))              # (1, 1)

learning_rate = 1.0
epochs = 5000

loss_history_mlp = []
accuracy_history_mlp = []

print(f"\nNetwork structure:")
print(f"  Input layer:  2 neurons")
print(f"  Hidden layer: 2 neurons (with sigmoid)")
print(f"  Output layer: 1 neuron (with sigmoid)")
print(f"\nTotal parameters: {W1.size + b1.size + W2.size + b2.size}")


for epoch in range(epochs): 
    # forward pass 
    # layer 1 
    z1 = X @ W1 + b1 # (4, 2) @ (2, 2) = (4, 2)
    a1 = sigmoid(z1) # (4, 2)

    # layer 2
    z2 = a1 @ W2 + b2 # (4, 2) @ (2, 1) = (4, 1)
    a2 = sigmoid(z2) # (4, 1)

    # loss 
    epsilon = 1e-15
    a2_safe = np.clip(a2 epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(a2_safe) + (1 - y) * np.log(1 - a2_safe))