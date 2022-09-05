import numpy as np

def sigmoid(x):
  return 1/ (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1-x)

training_inputs = np.array([[0,0,1], 
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1) #to make random calculations a good practice
synaptic_weights = 2 *np.random.random((3,1)) - 1 

print("Random Synaptic weights: ")
print(synaptic_weights)

for iter in range (2000):
  input_layer = training_inputs

  outputs = sigmoid(np.dot(input_layer, synaptic_weights))

  print("outputs :")
  print(outputs)
  error = training_outputs - outputs
  adjustments = error * sigmoid_derivative(outputs)
  print("ajustments :")
  print(adjustments)

  synaptic_weights += np.dot(input_layer.T , adjustments)  
  print("Synaptic weights after training: ")
  print(synaptic_weights)

  print("outputs after training :")
  print(outputs)





