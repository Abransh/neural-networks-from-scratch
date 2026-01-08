import numpy as np 
import matplotlib.pyplot as plt

# adding some simple data set poonts : y = 2x + 1 

np.random.seed(42) # if you know why i used 42 here, you are def goated hahahhaha 
X = np.random.rand(100,1) * 10 # 100 random numbers between 0 and 10 , the argument for rand is (rows, columns)
y = 2 * X + 1 + 0.1 + np.random.randn(100,1) * 0.5 # HERE we are adding some noise to the data because lets be real, perfection never comes handly or i would say doesnt exist 
# randn gives us a bell curve of noise (most points close to the line, few far away) - more realistic


# visualising the data 

plt.scatter(X,y, alpha = 0.5)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Data Set")
plt.show()

# here we have a simple data set and its plotted with some noise -1.5 to 1.5

#now we move on to intializing the neuron 

w= np.random.randn(1,1) # weight (1x1 matrix )
b= np.random.randn(1,1) # bias (1x1 matrix)

# hyperparameters (its like settings we choose)

learning_rate = 0.01 # how big steps we take when learning
epochs = 100 # how many times we want to run the loop or you can say how many times we want to look at all the data

loss_history = [] # to store the loss at each epoch and plot later


print(f"starting training..")
print(f"intial wieghts: {w[0,0]:.4f}")
print(f"intial bias: {b[0,0]:.4f}")
print(f"Target is: w=2, b=1 (since our data is y = 2x + 1)")
print("-" * 50)