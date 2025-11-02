import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

class NeuralNetwork3D:
    def __init__(self):
        # Initialize weights randomly
        self.weights = np.array([-1.5, -1.0])  # Start far from target
        self.learning_rate = 0.1
        self.loss_history = []
        self.weight_history = [self.weights.copy()]
        
    def forward_pass(self, x):
        return np.dot(x, self.weights)
    
    def calculate_loss(self, prediction, true_value):
        return (prediction - true_value) ** 2
    
    def backward_pass(self, x, prediction, true_value):
        error = prediction - true_value
        gradients = 2 * error * x
        return gradients
    
    def update_weights(self, gradients):
        self.weights = self.weights - self.learning_rate * gradients
        self.weight_history.append(self.weights.copy())

def create_3d_loss_landscape():
    """Create a 3D surface showing the loss landscape"""
    # True function we want to learn: y = 2*w1 + 3*w2
    # Loss = (prediction - true)^2 for all training examples
    
    # Create a grid of weight values
    w1_range = np.linspace(-3, 4, 50)
    w2_range = np.linspace(-2, 5, 50)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    
    # Training data
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
    y_true = np.array([8.0, 7.0, 9.0, 11.0])
    
    # Calculate loss for each weight combination
    Loss = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            total_loss = 0
            for k in range(len(X)):
                prediction = X[k, 0] * W1[i, j] + X[k, 1] * W2[i, j]
                loss = (prediction - y_true[k]) ** 2
                total_loss += loss
            Loss[i, j] = total_loss / len(X)  # Average loss
    
    return W1, W2, Loss, w1_range, w2_range

def plot_3d_backpropagation():
    """Create comprehensive 3D visualization"""
    
    # Create the neural network and train it
    nn = NeuralNetwork3D()
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
    y_true = np.array([8.0, 7.0, 9.0, 11.0])
    
    # Train for a few epochs
    epochs = 30
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X)):
            prediction = nn.forward_pass(X[i])
            loss = nn.calculate_loss(prediction, y_true[i])
            gradients = nn.backward_pass(X[i], prediction, y_true[i])
            nn.update_weights(gradients)
            epoch_loss += loss
        nn.loss_history.append(epoch_loss / len(X))
    
    # Create 3D loss landscape
    W1, W2, Loss, w1_range, w2_range = create_3d_loss_landscape()
    
    # Convert weight history to arrays for plotting
    weight_history = np.array(nn.weight_history)
    w1_path = weight_history[:, 0]
    w2_path = weight_history[:, 1]
    
    # Calculate the loss at each point in the path
    path_losses = []
    for w1, w2 in weight_history:
        total_loss = 0
        for i in range(len(X)):
            prediction = X[i, 0] * w1 + X[i, 1] * w2
            loss = (prediction - y_true[i]) ** 2
            total_loss += loss
        path_losses.append(total_loss / len(X))
    
    # Create the figure with 3D subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: 3D Loss Landscape with Training Path
    ax1 = fig.add_subplot(231, projection='3d')
    surf = ax1.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.6, 
                          linewidth=0, antialiased=True)
    
    # Plot the optimization path
    ax1.plot(w1_path, w2_path, path_losses, 'r-', linewidth=3, label='Training Path')
    ax1.scatter(w1_path, w2_path, path_losses, c=range(len(w1_path)), 
               cmap='plasma', s=50, depthshade=False)
    
    ax1.set_xlabel('Weight 1 (w1)')
    ax1.set_ylabel('Weight 2 (w2)')
    ax1.set_zlabel('Loss')
    ax1.set_title('3D Loss Landscape with Backpropagation Path\n(Red line shows weight updates)')
    ax1.legend()
    
    # Plot 2: Top-Down View (Weight Space)
    ax2 = fig.add_subplot(232)
    contour = ax2.contourf(W1, W2, Loss, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax2, label='Loss')
    
    # Plot the optimization path
    ax2.plot(w1_path, w2_path, 'r-', linewidth=2, label='Training Path')
    ax2.scatter(w1_path, w2_path, c=range(len(w1_path)), cmap='plasma', s=30)
    ax2.scatter([2.0], [3.0], color='green', s=200, marker='*', 
               label='Optimal Weights (2.0, 3.0)', edgecolors='white')
    ax2.scatter([w1_path[0]], [w2_path[0]], color='blue', s=100, marker='o', 
               label='Start Weights', edgecolors='white')
    
    ax2.set_xlabel('Weight 1 (w1)')
    ax2.set_ylabel('Weight 2 (w2)')
    ax2.set_title('Top-Down View: Weight Space\n(Color shows loss, path shows learning)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Reduction Over Time
    ax3 = fig.add_subplot(233)
    epochs_range = range(len(nn.loss_history))
    ax3.plot(epochs_range, nn.loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax3.set_xlabel('Training Epoch')
    ax3.set_ylabel('Average Loss')
    ax3.set_title('Loss Reduction During Training\n(How error decreases over time)')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale to see details
    
    # Plot 4: Weight Convergence
    ax4 = fig.add_subplot(234)
    epochs_weights = range(len(weight_history))
    ax4.plot(epochs_weights, w1_path, 'r-', linewidth=2, label='Weight 1 (w1)')
    ax4.plot(epochs_weights, w2_path, 'b-', linewidth=2, label='Weight 2 (w2)')
    ax4.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Target w1 = 2.0')
    ax4.axhline(y=3.0, color='blue', linestyle='--', alpha=0.7, label='Target w2 = 3.0')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Weight Value')
    ax4.set_title('Weight Convergence to Optimal Values\n(Lines approach dashed targets)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: 3D Path Only (Clean view)
    ax5 = fig.add_subplot(235, projection='3d')
    
    # Create a colormap for the path
    colors = range(len(w1_path))
    sc = ax5.scatter(w1_path, w2_path, path_losses, c=colors, cmap='plasma', 
                    s=50, depthshade=False)
    
    # Connect points with lines
    ax5.plot(w1_path, w2_path, path_losses, 'gray', alpha=0.5, linewidth=1)
    
    # Mark important points
    ax5.scatter([2.0], [3.0], [0], color='green', s=200, marker='*', 
               label='Optimal Point', edgecolors='white')
    ax5.scatter([w1_path[0]], [w2_path[0]], [path_losses[0]], color='red', s=100, 
               marker='o', label='Start Point', edgecolors='white')
    
    ax5.set_xlabel('Weight 1 (w1)')
    ax5.set_ylabel('Weight 2 (w2)')
    ax5.set_zlabel('Loss')
    ax5.set_title('3D Learning Path\n(Color shows training progress)')
    ax5.legend()
    
    # Plot 6: Gradient Vectors
    ax6 = fig.add_subplot(236)
    
    # Plot contour
    contour = ax6.contour(W1, W2, Loss, levels=20, colors='gray', alpha=0.6)
    
    # Plot some gradient vectors along the path
    for i in range(0, len(w1_path)-1, 3):
        dx = w1_path[i+1] - w1_path[i]
        dy = w2_path[i+1] - w2_path[i]
        ax6.arrow(w1_path[i], w2_path[i], dx, dy, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    ax6.plot(w1_path, w2_path, 'r-', linewidth=2, alpha=0.7, label='Learning Path')
    ax6.scatter([2.0], [3.0], color='green', s=100, marker='*', label='Optimal')
    ax6.set_xlabel('Weight 1 (w1)')
    ax6.set_ylabel('Weight 2 (w2)')
    ax6.set_title('Gradient Descent Direction\n(Arrows show weight updates)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final results
    print("🎯 TRAINING RESULTS:")
    print(f"Starting weights: [{nn.weight_history[0][0]:.3f}, {nn.weight_history[0][1]:.3f}]")
    print(f"Final weights: [{nn.weights[0]:.3f}, {nn.weights[1]:.3f}]")
    print(f"Target weights: [2.000, 3.000]")
    print(f"Final loss: {nn.loss_history[-1]:.6f}")
    print(f"Distance to target: {np.sqrt((nn.weights[0]-2.0)**2 + (nn.weights[1]-3.0)**2):.6f}")

# Run the 3D visualization
if __name__ == "__main__":
    plot_3d_backpropagation()