import numpy as np 
import matplotlib.pyplot as plt 

np.random.seed(42)

X= np.array([[0,0], [0,1], [1,0], [1,1]])

y= np.array([[0],[1],[1],[0]])

print("XOR Truth Table:")
print("Input 1 | Input 2 | Output")
print("-" * 30)
for i in range(4): 
    print(f"  {X[i, 0]:.0f}    |   {X[i, 1]:.0f}    |   {y[i, 0]:.0f}")

# lets visualise the XOR problem 

plt.figure(figsize= (8, 6))
colors = ['blue' if label == 0 else 'red' for label in y.flatten()]
for i in range(4):
    plt.scatter(X[i,0], X[i,1], c=colors[i], s = 200, edgecolors='k', linewidth=2, alpha = 0.7)
    plt.text(X[i,0] + 0.05, X[i,1] + 0.05,  f"({X[i, 0]:.0f},{X[i, 1]:.0f})", fontsize=12)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('XOR Problem: Can ONE line separate blue from red?')
plt.grid(True, alpha=0.3)
plt.legend(['Class 0', 'Class 1'])
plt.show()

#sigmoid function below
def sigmoid(z): 
    return 1/ (1 + np.exp(-z))

print("\n" + "="*60)
print("ATTEMPT 1: Single Neuron (This Will Fail!)")
print("="*60)    


w = np.random.randn(2, 1) * 0.5 
b= np.random.randn(1, 1) * 0.5 

learning_rate = 0.5
epochs = 5000

best_accuracy = []
loss_history = []


print(f"\nDebug - Shapes:")
print(f"X shape: {X.shape}")
print(f"w shape: {w.shape}")
print(f"b shape: {b.shape}")
print()

for epoch in range(epochs):
    # Check shapes at the START of each iteration
    if epoch < 3:  # Debug first 3 iterations
        print(f"\n=== Epoch {epoch} START ===")
        print(f"b shape at start: {b.shape}, value: {b}")
    
    # Forward pass
    z = X @ w + b
    y_pred = sigmoid(z)
    
    # Loss
    epsilon = 1e-15
    y_pred_safe = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(y_pred_safe) + (1 - y) * np.log(1 - y_pred_safe))
    loss_history.append(loss)
    
    # Accuracy
    predictions = (y_pred > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    best_accuracy.append(float(accuracy))
    
    # Gradients
    dz = y_pred - y
    dw = (X.T @ dz) / len(X)
    db = np.sum(dz, keepdims=True) / len(X)
    
    if epoch < 3:
        print(f"dw shape: {dw.shape}")
        print(f"db shape: {db.shape}, value: {db}")
    
    # Update
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    if epoch < 3:
        print(f"b shape AFTER update: {b.shape}, value: {b}")
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.1f}%")

print(f"\nBest accuracy achieved: {max(best_accuracy)*100:.1f}%")
print(f"Final predictions:")
for i in range(4):
    pred = sigmoid(X[i:i+1] @ w + b)[0, 0]
    pred_class = 1 if pred > 0.5 else 0
    print(f"  Input: {X[i]} → Predicted: {pred:.3f} (class {pred_class}) | True: {y[i, 0]}")

print("\n❌ Single neuron gets stuck at ~50% accuracy!")
print("It can only learn: 'always predict 0' or 'always predict 1'")
print("or guess randomly - it CANNOT learn XOR!")



# go to 04_multi_layer_network.py for the solution and how we attempt to solve this XOR issue using backpropogation and multi layer networks
# read xo_problem.md for mathematical analysis
