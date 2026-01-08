import numpy as np 
import matplotlib.pyplot as plt 

np.random.seed(42)

# Create binary classification dataset
# Class 0: Points around (2, 2)
# Class 1: Points around (6, 6)

n_samples = 100 

# class 0 (label = 0)

X_class0 = np.random.randn(n_samples // 2, 2) + np.array([2,2])
Y_class0 = np.zeros((n_samples // 2, 1))

# class 1 (label = 1)
X_class1 = np.random.randn(n_samples // 2, 2) + np.array([6,6])
Y_class1 = np.ones((n_samples // 2, 1))

# combine
X = np.vstack((X_class0, X_class1)) # (100, 2) - 2 features now!
Y = np.vstack((Y_class0, Y_class1)) # (100, 1) - binary labels

# shuffle 

indices = np.random.permutation(n_samples)
X = X[indices]
Y = Y[indices]


# visualizing the data
plt.figure(figsize=(8, 6))
plt.scatter(X[Y.flatten() == 0][:, 0], X[Y.flatten() == 0][:, 1], 
            c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X[Y.flatten() == 1][:, 0], X[Y.flatten() == 1][:, 1], 
            c='red', label='Class 1', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Classification Dataset')
plt.legend()
plt.grid(True)
plt.show()


# defining sigmoid activation function 
def sigmoid(z):
    """squashes any number into the range [0, 1]"""
    return 1 / (1 + np.exp(-z))


# Test sigmoid function
test_values = np.array([-10, -5, -1, 0, 1, 5, 10])
print("Testing Sigmoid:")
print("Input  | Output")
print("-" * 20)
for val in test_values:
    print(f"{val:6.1f} | {sigmoid(val):.6f}")    


#  Initialize Neuron (Now with 2 inputs!)
# Now we need 2 weights (one for each feature)
w = np.random.randn(2, 1)    # (2, 1) - 2 weights
b = np.random.randn(1, 1)    # (1, 1) - 1 bias

print("\nInitial parameters:")
print(f"w1 (weight for feature 1): {w[0, 0]:.4f}")
print(f"w2 (weight for feature 2): {w[1, 0]:.4f}")
print(f"b  (bias):                 {b[0, 0]:.4f}")


# hyperparameters

learning_rate = 0.1  # Slightly higher for classification
epochs = 1000 


# history tracking 

loss_history = []
accuracy_history = []


# The Training Loop (With New Loss Function!)
print("\nStarting training...")
print("-" * 60)

for epoch in range(epochs): 
    #1 forward pass: make predictions
    z = X @ w + b       # Linear combination: z = w1*x1 + w2*x2 + b
    y_pred= sigmoid(z)  # Apply sigmoid: squash to [0, 1]

    # 2. CALCULATE LOSS: Binary Cross-Entropy 

    epislon = 1e-15 # Small number to avoid log(0)
    y_pred_safe = np.clip(y_pred, epislon, 1-epislon)
    loss = -np.mean(Y * np.log(y_pred_safe) + (1-Y) * np.log(1-y_pred_safe))
    loss_history.append(loss)

    # Calculate accuracy
    predictions = (y_pred > 0.5).astype(int)  # Round: >0.5 → 1, else → 0
    accuracy = np.mean(predictions == Y)
    accuracy_history.append(accuracy)

    # 3. CALCULATE GRADIENTS

    dz = y_pred - Y          # Derivative of sigmoid + BCE is beautifully simple!
    dw = (1/len(X)) * (X.T @ dz)  # (2, 100) @ (100, 1) = (2, 1)
    db = (1/len(X)) * np.sum(dz)

    # 4. UPDATE PARAMETERS
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.1f}%")

print("-" * 60)
print(f"\nFinal Results:")
print(f"Loss:     {loss:.4f}")
print(f"Accuracy: {accuracy*100:.1f}%")
print(f"\nLearned weights:")
print(f"w1: {w[0, 0]:.4f}")
print(f"w2: {w[1, 0]:.4f}")
print(f"b:  {b[0, 0]:.4f}")
