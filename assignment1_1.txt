#======================================================================
# Assignment No.1: Back Propagation Algorithm
# Name: 
#======================================================================
print ('Back Propagation algorithm in Python')
import numpy as np
# RELU activation function
def leakyReLU(x,alpha):
    return max(alpha * z,z)
    
def ReLU_derivative(z,alpha):
    return 1 if z > 0 else alpha

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T
np.random.seed(1)
synaptic_weights = 2*np.random.random((3,1))-1
print('Random Starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(100):
    input_layer = training_inputs
    outputs=ReLU(np.dot(input_layer,synaptic_weights))
    error=training_outputs-outputs
    adjustments=error*ReLU_derivative()(outputs)
    synaptic_weights=synaptic_weights + np.dot(input_layer.T,adjustments)

print('Synaptic_weights after training:')
print(synaptic_weights)

print('Outputs after training:')
print(outputs)
