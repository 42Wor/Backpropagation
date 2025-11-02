import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        # Backward propagation
        m = X.shape[0]  # Number of samples
        
        # Calculate gradients
        # Output layer error
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer error
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def compute_loss(self, y_true, y_pred):
        # Binary cross-entropy loss
        m = y_true.shape[0]
        loss = -(1/m) * np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss
    
    def train(self, X, y, epochs):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, output)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Generate sample data
def generate_data(n_samples=1000):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    # Create a simple classification problem (XOR-like)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    y = y.reshape(-1, 1)
    return X, y

# Main execution
if __name__ == "__main__":
    # Generate data
    X, y = generate_data(1000)
    print(X, y)
    
    # Create and train network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    
    print("Training neural network...")
    losses = nn.train(X, y, epochs=10000)
    
    # Test the network
    test_input = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    predictions = nn.predict(test_input)
    
    print("\nTest predictions:")
    for i, inp in enumerate(test_input):
        print(f"Input: {inp} -> Prediction: {predictions[i][0]}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()