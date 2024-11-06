import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize inputs (i1, i2) and target outputs (for o1 and o2)
inputs = np.array([[0.05, 0.10]])
targets = np.array([[0.01, 0.82]])

# Initialize weights randomly for a 2-2-2 neural network (2 inputs, 2 hidden nodes, 2 outputs)
weights_input_hidden = np.array([[0.15, 0.25], [0.20, 0.30]])  # 2x2 weight matrix for input to hidden layer
weights_hidden_output = np.array([[0.40, 0.50], [0.45, 0.55]])  # 2x2 weight matrix for hidden to output layer

# Learning rate (can be varied)
learning_rate = 0.5

# Number of iterations
iterations = 10000

# Initialize a list to store the data for each iteration
data = []

# Training process
for iteration in range(iterations):
    # ** Forward Propagation **
    
    # Hidden layer activations
    hidden_input = np.dot(inputs, weights_input_hidden)  # Input to hidden layer (dot product of inputs and weights)
    hidden_output = sigmoid(hidden_input)                # Output of hidden layer after applying sigmoid activation
    
    # Output layer activations
    final_input = np.dot(hidden_output, weights_hidden_output)  # Input to output layer (dot product of hidden output and weights)
    final_output = sigmoid(final_input)                         # Output of output layer after applying sigmoid
    
    # ** Error Calculation **
    output_errors = 0.5 * (targets - final_output) ** 2  # Squared error
    total_error = np.sum(output_errors)                  # Total error for this iteration
    
    # Store data in a list for the current iteration
    data.append([iteration, total_error, final_output[0, 0], final_output[0, 1]])
    
    # ** Backpropagation **
    
    # Calculate output layer gradients
    output_delta = (final_output - targets) * sigmoid_derivative(final_output)
    
    # Calculate hidden layer gradients
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    
    # ** Update Weights **
    
    # Update weights for hidden-output connections
    weights_hidden_output -= learning_rate * np.dot(hidden_output.T, output_delta)
    
    # Update weights for input-hidden connections
    weights_input_hidden -= learning_rate * np.dot(inputs.T, hidden_delta)

# Convert the data into a Pandas DataFrame for tabular format
columns = ['Iteration', 'Total Error', 'Output o1', 'Output o2']
df = pd.DataFrame(data, columns=columns)

# Display the first few rows of the dataframe
print(df.head())

# Save the dataframe to a CSV file
df.to_csv('neural_network_training_log.csv', index=False)

# ** Plotting the Error Curve **

# Plot total error over iterations
plt.figure(figsize=(10, 6))
plt.plot(df['Iteration'], df['Total Error'], label='Total Error')
plt.title('Error Curve Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Total Error')
plt.grid(True)
plt.legend()
plt.show()
