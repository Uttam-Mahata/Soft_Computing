import numpy as np
import matplotlib.pyplot as plt
import csv


x1 = 0.1
x2 = 0.3
y_target = 0.03


w1 = 0.5
w2 = 0.2
b = 1.83


learning_rate = 0.001


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calculate_error(y_pred, y_true):
    return 0.5 * (y_pred - y_true) ** 2


def forward_propagation(w1, w2, b, x1, x2):
    z = w1 * x1 + w2 * x2 + b
    y_pred = sigmoid(z)
    return y_pred, z


def backward_propagation(y_pred, y_true, z, x1, x2, w1, w2, b, learning_rate):
    g_prime = y_pred * (1 - y_pred)
    dz = (y_pred - y_true) * g_prime
    
    
    dw1 = dz * x1
    dw2 = dz * x2
    db = dz
    
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b -= learning_rate * db
    
    return w1, w2, b, dw1, dw2, db

filename = "iterations_data.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(["Iteration", "w1", "w2", "x1", "x2", "b", "Error", "GradientError_w1", "GradientError_w2", "GradientError_b", "LearningRate"])

    errors = []
    iterations = []
    learning_rates = []

    for i in range(2):
        y_pred, z = forward_propagation(w1, w2, b, x1, x2)
        error = calculate_error(y_pred, y_target)
        w1, w2, b, dw1, dw2, db = backward_propagation(y_pred, y_target, z, x1, x2, w1, w2, b, learning_rate)
        
        writer.writerow([i+1, w1, w2, x1, x2, b, error, dw1, dw2, db, learning_rate])
        
        errors.append(error)
        iterations.append(i+1)
        learning_rates.append(learning_rate)
    
    convergence_threshold = 0.0001
    max_iterations = 10000
    current_iteration = 2

    while error > convergence_threshold and current_iteration < max_iterations:
        learning_rate = min(learning_rate * 1.1, 0.01)  

        y_pred, z = forward_propagation(w1, w2, b, x1, x2)
        error = calculate_error(y_pred, y_target)
        w1, w2, b, dw1, dw2, db = backward_propagation(y_pred, y_target, z, x1, x2, w1, w2, b, learning_rate)

        writer.writerow([current_iteration + 1, w1, w2, x1, x2, b, error, dw1, dw2, db, learning_rate])

        errors.append(error)
        iterations.append(current_iteration + 1)
        learning_rates.append(learning_rate)

        current_iteration += 1

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Error', color=color)
ax1.plot(iterations, errors, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Learning Rate', color=color)
ax2.plot(iterations, learning_rates, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("Error and Learning Rate vs. Iterations")
plt.show()

# Final weights, bias, and error
# w1, w2, b, error
