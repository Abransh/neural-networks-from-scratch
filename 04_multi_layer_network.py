# This is continuation of xor problem the previous file 03_xor_problem.py
import numpy as np 
import matplotlib.pyplot as plt 

print("\n" + "="*60)
print("ATTEMPT 2: Neural Network with Hidden Layer (This WILL Work!)")
print("="*60)

# network architecture 
# input(2) -> hidden layer ( 2 neurons ) -> output (1)

np.random.seed(42)

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
    a2_safe = np.clip(a2, epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(a2_safe) + (1 - y) * np.log(1 - a2_safe))
    loss_history_mlp.append(loss)

    # accuracy

    predictions = (a2 > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    accuracy_history_mlp.append(float(accuracy))


    # now we do backward pass or backpropogation

    dz2 = a2 - y
    dW2 = (a1.T @ dz2) / len(X) # (2, 4) @ (4, 1) = (2, 1)
    db2 = np.sum(dz2, keepdims = True , axis = 0) / len(X)

     # Hidden layer gradients (backprop through sigmoid)

    dz1 = (dz2 @ W2.T) *  a1 * (1 - a1) # (4, 1) @ (1, 2) * (4, 2) = (4, 2)
    dW1 = (X.T @ dz1) / len(X) # (2, 4) @ (4, 2) = (2, 2)
    db1 = np.sum(dz1, keepdims = True, axis = 0) / len(X)

    # updating the params 
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.1f}%")

print(f"\nFinal Loss: {loss:.4f}")
print(f"Final Accuracy: {accuracy*100:.1f}%")
print(f"\n✅ Neural network predictions:")
for i in range(4):
    # Forward pass for this input
    z1_test = X[i:i+1] @ W1 + b1
    a1_test = sigmoid(z1_test)
    z2_test = a1_test @ W2 + b2
    a2_test = sigmoid(z2_test)
    
    pred_class = 1 if a2_test[0, 0] > 0.5 else 0
    correct = "✓" if pred_class == y[i, 0] else "✗"
    print(f"  Input: {X[i]} → Predicted: {a2_test[0, 0]:.3f} (class {pred_class}) | True: {y[i, 0]} {correct}")

print("\n SUCCESS! The neural network learned XOR!")

# part 2 what did neurons learn
print("\n" + "="*60)
print("HIDDEN LAYER ANALYSIS: What did each neuron learn?")
print("="*60)

print("\nHidden neuron activations for each input:")
print("Input    | Hidden1 | Hidden2 | Output | Expected")
print("-" * 55)

for i in range(4):
    z1_test = X[i:i+1] @ W1 + b1
    a1_test = sigmoid(z1_test)
    z2_test = a1_test @ W2 + b2
    a2_test = sigmoid(z2_test)
    
    print(f"{X[i]}  |  {a1_test[0,0]:.3f}  |  {a1_test[0,1]:.3f}  |  {a2_test[0,0]:.3f}  |    {y[i,0]}")

print("\nInterpretation:")
print("Hidden Neuron 1 learns: 'Is at least one input 1?' (OR gate)")
print("Hidden Neuron 2 learns: 'Are both inputs 1?' (AND gate)")
print("Output combines them to get XOR!")

# part 3 visualise the decision boundary


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