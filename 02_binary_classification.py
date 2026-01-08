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