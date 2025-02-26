import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Example input (flattened image vectors)
X = np.array([[0.25, 0.5, 0.75, 1.0],  # Dog image example (simplified)
              [0.1, 0.3, 0.5, 0.7]])  # Cat image example (simplified)

# Example output (1 for dog, 0 for cat)
y = np.array([[1], [0]])

# Network architecture
input_layer_size = 4  # Number of features
hidden_layer1_size = 7  # First hidden layer neurons
hidden_layer2_size = 5  # Second hidden layer neurons
hidden_layer3_size = 3  # Third hidden layer neurons
output_layer_size = 1  # Binary classification

# Initialize random weights and biases
np.random.seed(0)  # For reproducibility
W1 = np.random.randn(input_layer_size, hidden_layer1_size)
b1 = np.random.randn(1, hidden_layer1_size)
W2 = np.random.randn(hidden_layer1_size, hidden_layer2_size)
b2 = np.random.randn(1, hidden_layer2_size)
W3 = np.random.randn(hidden_layer2_size, hidden_layer3_size)
b3 = np.random.randn(1, hidden_layer3_size)
W4 = np.random.randn(hidden_layer3_size, output_layer_size)
b4 = np.random.randn(1, output_layer_size)

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = tanh(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = tanh(z3)

    z4 = np.dot(a3, W4) + b4
    a4 = sigmoid(z4)

    # Compute error (MAE)
    error = y - a4
    loss = np.mean(np.abs(error))

    # Backpropagation
    d_a4 = error * sigmoid_derivative(a4)
    d_W4 = np.dot(a3.T, d_a4) * learning_rate
    d_b4 = np.sum(d_a4, axis=0, keepdims=True) * learning_rate

    d_a3 = np.dot(d_a4, W4.T) * tanh_derivative(a3)
    d_W3 = np.dot(a2.T, d_a3) * learning_rate
    d_b3 = np.sum(d_a3, axis=0, keepdims=True) * learning_rate

    d_a2 = np.dot(d_a3, W3.T) * tanh_derivative(a2)
    d_W2 = np.dot(a1.T, d_a2) * learning_rate
    d_b2 = np.sum(d_a2, axis=0, keepdims=True) * learning_rate

    d_a1 = np.dot(d_a2, W2.T) * tanh_derivative(a1)
    d_W1 = np.dot(X.T, d_a1) * learning_rate
    d_b1 = np.sum(d_a1, axis=0, keepdims=True) * learning_rate

    # Update weights and biases
    W4 += d_W4
    b4 += d_b4
    W3 += d_W3
    b3 += d_b3
    W2 += d_W2
    b2 += d_b2
    W1 += d_W1
    b1 += d_b1

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Final predictions
y_pred = a4
print("Final Predictions:", y_pred)

# Round the required values
a4_rounded = np.round(a4, 3)
a3_min_rounded = np.round(np.min(a3), 3)
W4_max_rounded = np.round(np.max(W4), 2)
W3_min_rounded = np.round(np.min(W3), 2)
loss_rounded = np.round(loss, 5)

# Print the rounded values
print(f"\na4 = {a4_rounded.tolist()}")
print(f"a3.min() = {a3_min_rounded}")
print(f"W4.max() = {W4_max_rounded}")
print(f"W3.min() = {W3_min_rounded}")
print(f"Loss after 10000 epochs: {loss_rounded}")

# General conclusion
if y_pred[0] > 0.5 and y_pred[1] < 0.5:
    print("\nNN predicts image of dog")
elif y_pred[0] < 0.5 and y_pred[1] > 0.5:
    print("\nNN predicts image of cat")
else:
    print("\nNN can't define correct image class")

#     Epoch 0, Loss: 0.509569
# Epoch 1000, Loss: 0.014816
# Epoch 2000, Loss: 0.006079
# Epoch 3000, Loss: 0.003917
# Epoch 4000, Loss: 0.002903
# Epoch 5000, Loss: 0.002309
# Epoch 6000, Loss: 0.001917
# Epoch 7000, Loss: 0.001639
# Epoch 8000, Loss: 0.001432
# Epoch 9000, Loss: 0.001271
# Final Predictions: [[0.99872171]
#  [0.00100799]]

# a4 = [[0.999], [0.001]]
# a3.min() = -1.0
# W4.max() = 3.83
# W3.min() = -3.19
# Loss after 10000 epochs: 0.00114
# NN predicts image of dog
