import numpy as np

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Sample input data: 12 records, 4 features
X = np.random.rand(12, 4)  # You can replace this with actual data
y = np.random.randint(0, 2, (12, 1))  # Binary target values (0 or 1)

# Add bias column (x0 = 1)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Now X has 5 features (x0 to x4)

# Initialize weights
np.random.seed(42)
weights_input_hidden = np.random.rand(5, 2)  # (x0–x4) to (A, B)
weights_hidden_output = np.random.rand(3, 1)  # (bias + A + B) to Z

# Training parameters
learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    net_hidden = np.dot(X, weights_input_hidden)
    out_hidden = sigmoid(net_hidden)
    out_hidden_bias = np.hstack([np.ones((out_hidden.shape[0], 1)), out_hidden])  # Add bias

    net_output = np.dot(out_hidden_bias, weights_hidden_output)
    out_output = sigmoid(net_output)

    # Loss
    loss = np.mean((y - out_output) ** 2)

    # Backward pass
    delta_output = (y - out_output) * sigmoid_derivative(out_output)
    delta_hidden = delta_output.dot(weights_hidden_output[1:].T) * sigmoid_derivative(out_hidden)

    # Weight updates
    weights_hidden_output += learning_rate * out_hidden_bias.T.dot(delta_output)
    weights_input_hidden += learning_rate * X.T.dot(delta_hidden)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal output predictions:")
print(np.round(out_output, 3))
