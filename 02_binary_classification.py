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
plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], 
            c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], 
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