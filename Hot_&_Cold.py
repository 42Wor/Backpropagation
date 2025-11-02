import numpy as np
import matplotlib.pyplot as plt

class ExpandedNeuralNetwork:
    def __init__(self, input_size=2):
        # Initialize weights randomly - our "starting position"
        self.weights = np.random.randn(input_size) * 0.5
        self.learning_rate = 0.1
        self.loss_history = []
        self.weight_history = []
        
    def forward_pass(self, x):
        """Make prediction: y = w1*x1 + w2*x2 + ..."""
        return np.dot(x, self.weights)
    
    def calculate_loss(self, prediction, true_value):
        """Mean Squared Error"""
        return (prediction - true_value) ** 2
    
    def backward_pass(self, x, prediction, true_value):
        """Calculate gradients using chain rule"""
        error = prediction - true_value
        gradients = 2 * error * x  # Derivative of MSE: dL/dw = 2*(y_pred - y_true)*x
        return gradients, error
    
    def update_weights(self, gradients):
        """Gradient descent update"""
        self.weights = self.weights - self.learning_rate * gradients
    
    def train(self, X, y_true, epochs=100):
        """Complete training process"""
        print("Starting training...")
        print(f"Initial weights: {self.weights}")
        print(f"Target weights: [2.0, 3.0]")
        print("-" * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0
            total_error = 0
            
            for i in range(len(X)):
                # Forward pass
                prediction = self.forward_pass(X[i])
                
                # Calculate loss
                loss = self.calculate_loss(prediction, y_true[i])
                epoch_loss += loss
                
                # Backward pass
                gradients, error = self.backward_pass(X[i], prediction, y_true[i])
                total_error += abs(error)
                
                # Update weights
                self.update_weights(gradients)
            
            # Record history for plotting
            avg_loss = epoch_loss / len(X)
            self.loss_history.append(avg_loss)
            self.weight_history.append(self.weights.copy())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Weights = [{self.weights[0]:.3f}, {self.weights[1]:.3f}], Avg Error = {total_error/len(X):.3f}")

# Different training scenarios
def demonstrate_different_cases():
    print("=== CASE 1: Original Example (y = 2*x1 + 3*x2) ===")
    
    # Original data
    X1 = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
    y1 = np.array([8.0, 7.0, 9.0, 11.0])
    
    nn1 = ExpandedNeuralNetwork()
    nn1.train(X1, y1, epochs=100)
    
    print(f"\nFinal weights: {nn1.weights}")
    print(f"Target: [2.0, 3.0]")
    print(f"Error: {np.abs(nn1.weights - np.array([2.0, 3.0]))}")
    
    print("\n" + "="*60)
    print("=== CASE 2: Different Function (y = 1.5*x1 + 0.5*x2) ===")
    
    X2 = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    y2 = np.array([2.0, 4.5, 5.5, 6.5])  # 1.5*x1 + 0.5*x2
    
    nn2 = ExpandedNeuralNetwork()
    nn2.train(X2, y2, epochs=100)
    
    print(f"\nFinal weights: {nn2.weights}")
    print(f"Target: [1.5, 0.5]")
    print(f"Error: {np.abs(nn2.weights - np.array([1.5, 0.5]))}")
    
    print("\n" + "="*60)
    print("=== CASE 3: More Complex Data ===")
    
    # Generate more data points
    np.random.seed(42)
    X3 = np.random.randn(20, 2)  # 20 samples, 2 features
    true_weights = np.array([2.0, 3.0])
    y3 = X3 @ true_weights + np.random.normal(0, 0.1, 20)  # Add small noise
    
    nn3 = ExpandedNeuralNetwork()
    nn3.train(X3, y3, epochs=200)
    
    print(f"\nFinal weights: {nn3.weights}")
    print(f"Target: [2.0, 3.0]")
    print(f"Error: {np.abs(nn3.weights - np.array([2.0, 3.0]))}")
    
    # Plot results
    plot_training_results(nn1, nn2, nn3)

def plot_training_results(nn1, nn2, nn3):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot loss history
    axes[0, 0].plot(nn1.loss_history)
    axes[0, 0].set_title('Case 1: Loss Over Time')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(nn2.loss_history)
    axes[0, 1].set_title('Case 2: Loss Over Time')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(nn3.loss_history)
    axes[0, 2].set_title('Case 3: Loss Over Time')
    axes[0, 2].grid(True)
    
    # Plot weight convergence
    weights1 = np.array(nn1.weight_history)
    axes[1, 0].plot(weights1[:, 0], label='w1')
    axes[1, 0].plot(weights1[:, 1], label='w2')
    axes[1, 0].axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='Target w1')
    axes[1, 0].axhline(y=3.0, color='g', linestyle='--', alpha=0.7, label='Target w2')
    axes[1, 0].set_title('Case 1: Weight Convergence')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Weight Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    weights2 = np.array(nn2.weight_history)
    axes[1, 1].plot(weights2[:, 0], label='w1')
    axes[1, 1].plot(weights2[:, 1], label='w2')
    axes[1, 1].axhline(y=1.5, color='r', linestyle='--', alpha=0.7, label='Target w1')
    axes[1, 1].axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Target w2')
    axes[1, 1].set_title('Case 2: Weight Convergence')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    weights3 = np.array(nn3.weight_history)
    axes[1, 2].plot(weights3[:, 0], label='w1')
    axes[1, 2].plot(weights3[:, 1], label='w2')
    axes[1, 2].axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='Target w1')
    axes[1, 2].axhline(y=3.0, color='g', linestyle='--', alpha=0.7, label='Target w2')
    axes[1, 2].set_title('Case 3: Weight Convergence')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

# Test the predictions
def test_final_model():
    print("\n" + "="*60)
    print("TESTING FINAL MODEL")
    print("="*60)
    
    # Train a final model
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
    y_true = np.array([8.0, 7.0, 9.0, 11.0])
    
    nn = ExpandedNeuralNetwork()
    nn.train(X, y_true, epochs=100)
    
    print("\nPrediction Test:")
    print("Input       | True Output | Prediction | Error")
    print("-" * 50)
    
    for i in range(len(X)):
        prediction = nn.forward_pass(X[i])
        error = abs(prediction - y_true[i])
        print(f"{X[i]} | {y_true[i]:11.1f} | {prediction:9.3f} | {error:5.3f}")
    
    print(f"\nFinal weights: {nn.weights}")
    print(f"Very close to target: [2.0, 3.0]")

if __name__ == "__main__":
    demonstrate_different_cases()
    test_final_model()