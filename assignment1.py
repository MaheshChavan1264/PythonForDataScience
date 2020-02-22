#======================================================================
# Assignment No.1: Back Propagation Algorithm
# Name: Mahesh Tatyasaheb Chavan
#======================================================================
print ('Back Propagation algorithm in Python')
import numpy as np
# RELU activation function
def ReLU(x):
    return np.maximum(0.0,x)

def ReLU_derivative(x):
    if x<=0:
        return 0
    else:
        return 1


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
