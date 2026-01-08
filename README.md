# neural-networks-from-scratch 

so i am building first one neuron in 01_single_neuron.py to plot 100 numbers between 0-10 and then adding some noise to it and then visualising it using matplotlib

the equation is y = 2x + 1 + 0.1 + np.random.randn(100,1) * 0.5 
or y = 2x + 1 + 0.1 + noise
for noise we use randn because it gives us a bell curve of noise (most points close to the line, few far away) - more realistic