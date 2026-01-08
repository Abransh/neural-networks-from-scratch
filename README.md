# neural-networks-from-scratch 

so i am building first one neuron in 01_single_neuron.py to plot 100 numbers between 0-10 and then adding some noise to it and then visualising it using matplotlib

the equation is y = 2x + 1 + 0.1 + np.random.randn(100,1) * 0.5 
or y = 2x + 1 + 0.1 + noise
for noise we use randn because it gives us a bell curve of noise (most points close to the line, few far away) - more realistic


for 2nd part we intialize the neuron with random weights and bias 
define learning rate and epochs 
define loss history


for 3rd part we define the forward pass 
we make the model run forward pass and calculate the loss 
we update the weights and bias and get closer to the line 

for 4th part its just visualizing the learning process 

I built a complete learning system:

Forward pass (make predictions)
Loss calculation (measure error)
Backpropagation (calculate gradients)
Parameter update (gradient descent)
  
in 01_single_neuron.pu 