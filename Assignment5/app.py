from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Neural network parameters from your provided code
input_size = 7
hidden_layer1_size = 10
hidden_layer2_size = 5
output_size = 1
learning_rate = 0.1
epochs = 1000

# Initialize weights and biases
np.random.seed(0)
W1 = np.random.randn(input_size, hidden_layer1_size) * 0.01
b1 = np.zeros((1, hidden_layer1_size))
W2 = np.random.randn(hidden_layer1_size, hidden_layer2_size) * 0.01
b2 = np.zeros((1, hidden_layer2_size))
W3 = np.random.randn(hidden_layer2_size, output_size) * 0.01
b3 = np.zeros((1, output_size))

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward propagation function
def forward_propagation(inputs):
    Z1 = np.dot(inputs, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Train the neural network
def train_network():
    global W1, b1, W2, b2, W3, b3
    inputs = np.array([
        [1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1]
    ])
    targets = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])

    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(inputs)
        dZ3 = A3 - targets
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        dZ2 = np.dot(dZ3, W3.T) * (A2 * (1 - A2))
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, W2.T) * (A1 * (1 - A1))
        dW1 = np.dot(inputs.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

train_network()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.json is None:
        return jsonify(error="No JSON data received"), 400
    segment_data = request.json.get('segments')
    if segment_data is None:
        return jsonify(error="'segments' key not found in JSON data"), 400
    _, _, _, _, _, prediction = forward_propagation(np.array([segment_data]))
    result = 1 if prediction[0][0] >= 0.5 else 0
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
