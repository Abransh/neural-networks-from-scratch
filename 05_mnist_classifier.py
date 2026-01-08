import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# run pip install scikit-learn  

print("Downloading MNIST dataset...")
# Download MNIST (this might take a minute first time)

mnist = fetch_openml('mnist_784', version = 1, parser = 'auto')
X, y = mnist.data.values , mnist.target.values.astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")

# visualising some data
fig, axes = plt.subplots(2, 5, figsize = (12, 5))
for i, ax in enumerate(axes.flat): 
    ax.imshow(X[i].reshape(28,28), cmap = "gray")
    ax.set_title(f'Label: {y[i]}')
    ax.axis("off")
plt.suptitle('Sample MNIST Digits', fontsize=16)
plt.tight_layout()    
plt.show() 


# preprocess
X = X / 255.0 

# split the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

