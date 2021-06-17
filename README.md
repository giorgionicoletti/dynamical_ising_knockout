Given a rasterplot of the response of a network of neurons to a repeated stimulus, the code reconstruct a dynamical asynchronous
and asymmetric Ising model with Glauber dynamics. A Hidden Markov Model with 2 hidden states (whether the stimulus is present or not)
is fitted to the original data and the reconstructed ones. Then, a single neuron is removed from the model and a new HMM is fitted to 
the simulated data. The resulting high-dimensional set of transition matrices and emission matrices is analyzed using appropriate distance
measures in the corresponding high-dimensional space.
