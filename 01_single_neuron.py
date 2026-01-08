import numpy as np 
import matplotlib.pyplot as plt

# adding some simple data set poonts : y = 2x + 1 

np.random.seed(42) # if you know why i used 42 here, you are def goated hahahhaha 
X = np.random.rand(100,1) * 10 # 100 random numbers between 0 and 10 
y = 2 * X + 1 + 0.1 + np.random.randn(100,1) * 0.5 

# visualising the data 

plt.scatter(X,y, alpha = 0.5)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Data Set")
plt.show()