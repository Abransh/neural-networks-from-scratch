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
colors = ['blue' if label == 0 else 'red' for lavel in y.flatten()]
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

