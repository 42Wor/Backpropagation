import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class InteractiveBackprop:
    def __init__(self):
        # Training data: y = 2*x1 + 3*x2
        self.X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
        self.y_true = np.array([8.0, 7.0, 9.0, 11.0])
        
        # Initialize weights
        self.initial_weights = np.array([-1.0, 0.5])
        self.weights = self.initial_weights.copy()
        self.learning_rate = 0.1
        
        # Training state
        self.current_sample = 0
        self.current_prediction = None
        self.current_loss = None
        self.current_gradients = None
        self.step_history = []
        
        # Create the loss landscape
        self.create_loss_landscape()
    
    def create_loss_landscape(self):
        """Create 3D loss landscape"""
        w1_range = np.linspace(-3, 4, 50)
        w2_range = np.linspace(-2, 5, 50)
        self.W1, self.W2 = np.meshgrid(w1_range, w2_range)
        
        self.Loss = np.zeros_like(self.W1)
        for i in range(self.W1.shape[0]):
            for j in range(self.W1.shape[1]):
                total_loss = 0
                for k in range(len(self.X)):
                    prediction = self.X[k, 0] * self.W1[i, j] + self.X[k, 1] * self.W2[i, j]
                    loss = (prediction - self.y_true[k]) ** 2
                    total_loss += loss
                self.Loss[i, j] = total_loss / len(self.X)
    
    def forward_pass(self, sample_idx=None):
        """Execute forward pass"""
        if sample_idx is None:
            sample_idx = self.current_sample
        
        x = self.X[sample_idx]
        self.current_prediction = np.dot(x, self.weights)
        return self.current_prediction
    
    def loss_calculation(self, sample_idx=None):
        """Calculate loss for current prediction"""
        if sample_idx is None:
            sample_idx = self.current_sample
        
        if self.current_prediction is None:
            self.forward_pass(sample_idx)
        
        self.current_loss = (self.current_prediction - self.y_true[sample_idx]) ** 2
        return self.current_loss
    
    def backward_pass(self, sample_idx=None):
        """Calculate gradients"""
        if sample_idx is None:
            sample_idx = self.current_sample
        
        if self.current_prediction is None:
            self.forward_pass(sample_idx)
        
        error = self.current_prediction - self.y_true[sample_idx]
        self.current_gradients = 2 * error * self.X[sample_idx]
        return self.current_gradients
    
    def weight_update(self):
        """Update weights using calculated gradients"""
        if self.current_gradients is None:
            self.backward_pass()
        
        old_weights = self.weights.copy()
        self.weights = self.weights - self.learning_rate * self.current_gradients
        
        # Record this step
        step_info = {
            'step_number': len(self.step_history),
            'sample': self.current_sample,
            'input': self.X[self.current_sample],
            'prediction': self.current_prediction,
            'true_value': self.y_true[self.current_sample],
            'loss': self.current_loss,
            'gradients': self.current_gradients.copy(),
            'weights_before': old_weights,
            'weights_after': self.weights.copy()
        }
        self.step_history.append(step_info)
        
        # Move to next sample
        self.current_sample = (self.current_sample + 1) % len(self.X)
        
        # Reset for next step
        self.current_prediction = None
        self.current_loss = None
        self.current_gradients = None
        
        return step_info
    
    def go_backward(self, steps=1):
        """Go backwards in training history - FIXED VERSION"""
        if len(self.step_history) == 0:
            print("❌ No history to go back to!")
            return None
        
        steps_to_go_back = min(steps, len(self.step_history))
        
        print(f"\n⏪ REWINDING {steps_to_go_back} STEP(S)")
        print("-" * 50)
        
        # Remove the last 'steps_to_go_back' steps from history
        removed_steps = self.step_history[-steps_to_go_back:]
        self.step_history = self.step_history[:-steps_to_go_back]
        
        if len(self.step_history) > 0:
            # Restore weights to the last step in remaining history
            last_step = self.step_history[-1]
            self.weights = last_step['weights_after'].copy()
            # Restore the correct sample for the NEXT step
            self.current_sample = (last_step['sample'] + 1) % len(self.X)
        else:
            # Back to beginning
            self.weights = self.initial_weights.copy()
            self.current_sample = 0
        
        # Print what we're undoing
        for i, step in enumerate(removed_steps):
            print(f"Undoing step {step['step_number'] + 1}:")
            print(f"  Sample: {step['sample'] + 1}")
            print(f"  Weights: [{step['weights_before'][0]:.3f}, {step['weights_before'][1]:.3f}] → "
                  f"[{step['weights_after'][0]:.3f}, {step['weights_after'][1]:.3f}]")
            print(f"  Loss: {step['loss']:.3f}")
            if i < len(removed_steps) - 1:
                print("  ---")
        
        print(f"✅ Now at step {len(self.step_history)}")
        return removed_steps
    
    def run_full_training_step(self, sample_idx=None):
        """Run all four phases for one sample"""
        if sample_idx is None:
            sample_idx = self.current_sample
        
        current_step_number = len(self.step_history)
        print(f"\n🎯 TRAINING STEP {current_step_number + 1}:")
        print(f"Sample {sample_idx + 1}: Input: {self.X[sample_idx]}, Target: {self.y_true[sample_idx]}")
        print("-" * 50)
        
        # 1. Forward Pass
        prediction = self.forward_pass(sample_idx)
        print(f"1. FORWARD PASS:")
        print(f"   Weights: [{self.weights[0]:.3f}, {self.weights[1]:.3f}]")
        print(f"   Prediction: {prediction:.3f}")
        
        # 2. Loss Calculation
        loss = self.loss_calculation(sample_idx)
        print(f"2. LOSS CALCULATION:")
        print(f"   Error: {prediction - self.y_true[sample_idx]:.3f}")
        print(f"   Loss: {loss:.3f}")
        
        # 3. Backward Pass
        gradients = self.backward_pass(sample_idx)
        print(f"3. BACKWARD PASS:")
        print(f"   Gradients: [{gradients[0]:.3f}, {gradients[1]:.3f}]")
        print(f"   Direction: {'decrease' if gradients[0] > 0 else 'increase'} w1, "
              f"{'decrease' if gradients[1] > 0 else 'increase'} w2")
        
        # 4. Weight Update
        step_info = self.weight_update()
        print(f"4. WEIGHT UPDATE:")
        print(f"   Old weights: [{step_info['weights_before'][0]:.3f}, {step_info['weights_before'][1]:.3f}]")
        print(f"   New weights: [{self.weights[0]:.3f}, {self.weights[1]:.3f}]")
        
        return step_info

def interactive_demo():
    """Interactive demonstration where you control backpropagation"""
    
    # Create the neural network
    nn = InteractiveBackprop()
    
    print("🧠 INTERACTIVE BACKPROPAGATION DEMO")
    print("=" * 60)
    print("Target function: y = 2*x1 + 3*x2")
    print(f"Initial weights: [{nn.weights[0]:.3f}, {nn.weights[1]:.3f}]")
    print(f"Target weights: [2.000, 3.000]")
    print("\nControls:")
    print("- Press 'n' for next training step")
    print("- Press 'r' to run multiple steps")
    print("- Press 'b' to go backwards (negative steps)")
    print("- Press 'p' to plot current state")
    print("- Press 's' to show current status")
    print("- Press 'q' to quit")
    print("=" * 60)
    
    # Create initial plot
    fig = plt.figure(figsize=(15, 10))
    
    def plot_current_state():
        plt.clf()
        
        # Convert weight history to arrays
        if len(nn.step_history) > 0:
            w1_history = [step['weights_after'][0] for step in nn.step_history]
            w2_history = [step['weights_after'][1] for step in nn.step_history]
            
            # Calculate losses for the path
            path_losses = []
            for w1, w2 in zip(w1_history, w2_history):
                total_loss = 0
                for i in range(len(nn.X)):
                    prediction = nn.X[i, 0] * w1 + nn.X[i, 1] * w2
                    loss = (prediction - nn.y_true[i]) ** 2
                    total_loss += loss
                path_losses.append(total_loss / len(nn.X))
        else:
            w1_history = [nn.weights[0]]
            w2_history = [nn.weights[1]]
            path_losses = [0]
        
        # Plot 1: 3D Loss Landscape
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot_surface(nn.W1, nn.W2, nn.Loss, cmap='viridis', alpha=0.6)
        
        if len(nn.step_history) > 0:
            ax1.plot(w1_history, w2_history, path_losses, 'r-', linewidth=3)
            ax1.scatter(w1_history, w2_history, path_losses, 
                       c=range(len(w1_history)), cmap='plasma', s=50)
        
        ax1.scatter([2.0], [3.0], [0], color='green', s=200, marker='*', 
                   label='Target [2,3]')
        ax1.set_xlabel('Weight 1')
        ax1.set_ylabel('Weight 2')
        ax1.set_zlabel('Loss')
        ax1.set_title(f'3D Loss Landscape (Step {len(nn.step_history)})')
        
        # Plot 2: Top-Down View
        ax2 = fig.add_subplot(232)
        contour = ax2.contourf(nn.W1, nn.W2, nn.Loss, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax2)
        
        if len(nn.step_history) > 0:
            ax2.plot(w1_history, w2_history, 'r-', linewidth=2)
            ax2.scatter(w1_history, w2_history, c=range(len(w1_history)), 
                       cmap='plasma', s=30)
        
        ax2.scatter([2.0], [3.0], color='green', s=100, marker='*')
        ax2.set_xlabel('Weight 1')
        ax2.set_ylabel('Weight 2')
        ax2.set_title('Weight Space')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Weight Progress
        ax3 = fig.add_subplot(233)
        if len(nn.step_history) > 0:
            steps = range(len(w1_history))
            ax3.plot(steps, w1_history, 'r-', label='Weight 1', marker='o')
            ax3.plot(steps, w2_history, 'b-', label='Weight 2', marker='s')
        else:
            # Show initial point
            ax3.scatter([0], [nn.weights[0]], color='red', s=50, label='Weight 1')
            ax3.scatter([0], [nn.weights[1]], color='blue', s=50, label='Weight 2')
            
        ax3.axhline(y=2.0, color='red', linestyle='--', label='Target w1')
        ax3.axhline(y=3.0, color='blue', linestyle='--', label='Target w2')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Weight Value')
        ax3.set_title('Weight Convergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Loss History
        ax4 = fig.add_subplot(234)
        if len(nn.step_history) > 0:
            losses = [step['loss'] for step in nn.step_history]
            ax4.plot(range(len(losses)), losses, 'g-', marker='o')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Sample Loss')
            ax4.set_title('Loss per Training Sample')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Current Predictions vs Targets
        ax5 = fig.add_subplot(235)
        current_predictions = [np.dot(x, nn.weights) for x in nn.X]
        samples = range(1, len(nn.X) + 1)
        ax5.bar([s - 0.2 for s in samples], nn.y_true, width=0.4, 
               label='True Values', alpha=0.7)
        ax5.bar([s + 0.2 for s in samples], current_predictions, width=0.4, 
               label='Predictions', alpha=0.7)
        ax5.set_xlabel('Sample')
        ax5.set_ylabel('Value')
        ax5.set_title('Current Predictions vs Targets')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Gradient History
        ax6 = fig.add_subplot(236)
        if len(nn.step_history) > 0:
            grad1 = [step['gradients'][0] for step in nn.step_history]
            grad2 = [step['gradients'][1] for step in nn.step_history]
            ax6.plot(range(len(grad1)), grad1, 'orange', label='Gradient w1', marker='o')
            ax6.plot(range(len(grad2)), grad2, 'purple', label='Gradient w2', marker='s')
            ax6.set_xlabel('Training Step')
            ax6.set_ylabel('Gradient Value')
            ax6.set_title('Gradient History')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def show_status():
        """Show current status"""
        current_step = len(nn.step_history)
        print(f"\n📊 CURRENT STATUS:")
        print(f"Current step: {current_step}")
        print(f"Next sample: {nn.current_sample + 1}")
        print(f"Current weights: [{nn.weights[0]:.3f}, {nn.weights[1]:.3f}]")
        print(f"Target weights: [2.000, 3.000]")
        print(f"Distance to target: {np.sqrt((nn.weights[0]-2.0)**2 + (nn.weights[1]-3.0)**2):.6f}")
        print(f"Steps in history: {len(nn.step_history)}")
        
        # Calculate current total loss
        total_loss = 0
        for i in range(len(nn.X)):
            prediction = np.dot(nn.X[i], nn.weights)
            loss = (prediction - nn.y_true[i]) ** 2
            total_loss += loss
        print(f"Current average loss: {total_loss/len(nn.X):.6f}")
    
    # Initial plot
    plot_current_state()
    show_status()
    
    # Interactive loop
    while True:
        command = input("\nEnter command (n/r/b/p/s/q): ").strip().lower()
        
        if command == 'n':
            # Next step
            nn.run_full_training_step()
            plot_current_state()
            show_status()
            
        elif command == 'r':
            # Run multiple steps
            try:
                num_steps = int(input("How many steps to run? "))
                if num_steps > 0:
                    for i in range(num_steps):
                        nn.run_full_training_step()
                    plot_current_state()
                    show_status()
                else:
                    print("❌ Please enter a positive number for forward steps")
            except ValueError:
                print("❌ Please enter a valid number")
                
        elif command == 'b':
            # Go backwards
            try:
                num_steps = int(input("How many steps to go back? "))
                if num_steps > 0:
                    nn.go_backward(num_steps)
                    plot_current_state()
                    show_status()
                else:
                    print("❌ Please enter a positive number for backward steps")
            except ValueError:
                print("❌ Please enter a valid number")
                
        elif command == 'p':
            # Plot current state
            plot_current_state()
            
        elif command == 's':
            # Show status
            show_status()
            
        elif command == 'q':
            # Quit
            print("\n🎉 FINAL RESULTS:")
            show_status()
            print(f"Total training steps taken: {len(nn.step_history)}")
            break
            
        else:
            print("❌ Unknown command. Use: n (next), r (run), b (back), p (plot), s (status), q (quit)")

# Run the interactive demo
if __name__ == "__main__":
    interactive_demo()