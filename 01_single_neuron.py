import numpy as np 
import matplotlib.pyplot as plt
#part 1
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
#part 2
#now we move on to intializing the neuron 

w= np.random.randn(1,1) # weight (1x1 matrix )
b= np.random.randn(1,1) # bias (1x1 matrix)

# hyperparameters (its like settings we choose)

learning_rate = 0.01 # how big steps we take when learning
epochs = 10000 # how many times we want to run the loop or you can say how many times we want to look at all the data

loss_history = [] # to store the loss at each epoch and plot later


print(f"starting training..")
print(f"intial wieghts: {w[0,0]:.4f}")
print(f"intial bias: {b[0,0]:.4f}")
print(f"Target is: w=2, b=1 (since our data is y = 2x + 1)")
print("-" * 50)


# now we move on to training the neuron 
#part 3
for epoch in range(epochs):
    # forward pass: making the predicitons
    y_pred = w * X + b 

    #2 CALCULATE LOSS: How wrong are we? (I am never wrong but python or neuron can be)

    loss= np.mean((y_pred - y)**2) # mean squared error
    loss_history.append(loss)
    
     # 3. CALCULATE GRADIENTS: Which direction to adjust w and b?
    dw = (2/len(X)) * np.sum((y_pred - y)*X) # partial derivative of loss with respect to w
    db = (2/len(X)) * np.sum((y_pred - y)) # partial derivative of loss with respect to b
    
    # 4 update the parameters: take a step in the direction of the gradient 
    
    w = w - learning_rate * dw 
    b = b - learning_rate * db 

    # pring progress 
    if epoch % 10 == 0 : 
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w[0,0]:.4f} | b: {b[0,0]:.4f}")

print("-" * 50)
print(f"Final weight: {w[0, 0]:.4f} (target: 2.0)")
print(f"Final bias: {b[0, 0]:.4f} (target: 1.0)")

#part 4 Visualize the Learning

# Plot 1: Loss going down over time
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time (Should Decrease!)')
plt.grid(True)

# Plot 2: Original data + learned line
plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, w * X + b, 'r-', linewidth=2, label=f'Learned: y = {w[0,0]:.2f}x + {b[0,0]:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Neuron Found the Pattern!')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()