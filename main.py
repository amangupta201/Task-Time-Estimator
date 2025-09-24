import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random


class NeuralNetwork:
    """
    A feedforward neural network built from scratch using only numpy
    For predicting task completion times based on user patterns
    """

    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, learning_rate: float = 0.001):
        """
        Initialize the neural network

        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes [64, 32] means 2 hidden layers
            output_size: Number of outputs (1 for our time prediction)
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layers)

        # Initialize weights and biases using He initialization (better for ReLU)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # He initialization for ReLU activation
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i])
            bias_vector = np.zeros((1, self.layers[i + 1]))

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def linear(self, x):
        """Linear activation (for output layer)"""
        return x

    def linear_derivative(self, x):
        """Derivative of linear activation"""
        return np.ones_like(x)

    def forward_propagation(self, X):
        """
        Forward pass through the network

        Args:
            X: Input data (batch_size, input_size)

        Returns:
            activations: List of activations for each layer
            z_values: List of pre-activation values for each layer
        """
        activations = [X]
        z_values = []

        current_activation = X

        for i in range(self.num_layers - 1):
            # Linear transformation
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            z_values.append(z)

            # Apply activation function
            if i == self.num_layers - 2:  # Output layer - use linear
                current_activation = self.linear(z)
            else:  # Hidden layers - use ReLU
                current_activation = self.relu(z)

            activations.append(current_activation)

        return activations, z_values

    def backward_propagation(self, X, y, activations, z_values):
        """
        Backward pass to compute gradients

        Args:
            X: Input data
            y: True labels
            activations: Activations from forward pass
            z_values: Pre-activation values from forward pass

        Returns:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        m = X.shape[0]  # Batch size

        weight_gradients = []
        bias_gradients = []

        # Initialize gradients
        for i in range(self.num_layers - 1):
            weight_gradients.append(np.zeros_like(self.weights[i]))
            bias_gradients.append(np.zeros_like(self.biases[i]))

        # Compute error for output layer (Mean Squared Error derivative)
        delta = activations[-1] - y  # For MSE loss

        # Backpropagate through all layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients for current layer
            weight_gradients[i] = np.dot(activations[i].T, delta) / m
            bias_gradients[i] = np.sum(delta, axis=0, keepdims=True) / m

            # Compute delta for previous layer (if not input layer)
            if i > 0:
                if i == self.num_layers - 2:  # Coming from output layer
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i - 1])
                else:  # Coming from hidden layer
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i - 1])

        return weight_gradients, bias_gradients

    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradients with gradient clipping"""
        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0

        for i in range(self.num_layers - 1):
            # Clip weight gradients
            weight_grad_norm = np.linalg.norm(weight_gradients[i])
            if weight_grad_norm > max_grad_norm:
                weight_gradients[i] = weight_gradients[i] * (max_grad_norm / weight_grad_norm)

            # Clip bias gradients
            bias_grad_norm = np.linalg.norm(bias_gradients[i])
            if bias_grad_norm > max_grad_norm:
                bias_gradients[i] = bias_gradients[i] * (max_grad_norm / bias_grad_norm)

            # Update parameters
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def compute_loss(self, y_pred, y_true):
        """Compute Mean Squared Error loss"""
        return np.mean((y_pred - y_true) ** 2)

    def train_step(self, X, y):
        """Single training step"""
        # Forward pass
        activations, z_values = self.forward_propagation(X)

        # Compute loss
        loss = self.compute_loss(activations[-1], y)

        # Backward pass
        weight_gradients, bias_gradients = self.backward_propagation(X, y, activations, z_values)

        # Update parameters
        self.update_parameters(weight_gradients, bias_gradients)

        return loss

    def predict(self, X):
        """Make predictions"""
        activations, _ = self.forward_propagation(X)
        return activations[-1]

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, batch_size=32, verbose=True):
        """
        Train the neural network

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Whether to print training progress
        """
        train_losses = []
        val_losses = []

        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                batch_loss = self.train_step(X_batch, y_batch)
                epoch_loss += batch_loss

            avg_epoch_loss = epoch_loss / n_batches
            train_losses.append(avg_epoch_loss)

            # Validation loss
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(val_pred, y_val)
                val_losses.append(val_loss)

            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss:.4f}")

        return train_losses, val_losses


class TaskTimeDataGenerator:
    """
    Generate realistic synthetic data for task time estimation
    Different user profiles with different estimation biases
    """

    def __init__(self, user_profile="balanced"):
        """
        Initialize data generator

        Args:
            user_profile: "optimistic", "realistic", "pessimistic", "balanced"
        """
        self.user_profile = user_profile
        self.task_categories = {
            0: "coding",
            1: "writing",
            2: "research",
            3: "meeting",
            4: "admin",
            5: "creative",
            6: "debugging"
        }

        # Base time multipliers for different categories (in minutes)
        self.category_base_times = {
            0: 60,  # coding
            1: 30,  # writing
            2: 45,  # research
            3: 25,  # meeting
            4: 15,  # admin
            5: 90,  # creative
            6: 120  # debugging
        }

        # User bias multipliers
        self.bias_multipliers = {
            "optimistic": 0.6,  # Underestimates by 40%
            "realistic": 1.0,  # Accurate estimates
            "pessimistic": 1.4,  # Overestimates by 40%
            "balanced": 1.0  # Mix of all types
        }

    def generate_task_features(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic task data

        Returns:
            X: Features [task_category, complexity, description_length, hour_of_day, energy_level]
            y: Actual completion times in minutes
        """
        np.random.seed(42)  # For reproducible results

        X = []
        y = []

        for _ in range(n_samples):
            # Task category (0-6)
            task_category = np.random.randint(0, 7)

            # Complexity (1-10)
            complexity = np.random.randint(1, 11)

            # Description length (proxy for task detail) (5-100 words)
            description_length = np.random.randint(5, 101)

            # Hour of day (affects productivity)
            hour_of_day = np.random.randint(8, 19)  # 8 AM to 6 PM

            # Energy level (1-10, affected by time of day)
            if 9 <= hour_of_day <= 11 or 14 <= hour_of_day <= 16:  # Peak hours
                energy_level = np.random.randint(7, 11)
            elif hour_of_day <= 9 or hour_of_day >= 17:  # Low energy hours
                energy_level = np.random.randint(3, 7)
            else:  # Moderate hours
                energy_level = np.random.randint(5, 9)

            # Calculate actual time based on realistic factors
            base_time = self.category_base_times[task_category]

            # Complexity affects time exponentially (not linearly)
            complexity_multiplier = 1 + (complexity - 5) * 0.3

            # Description length affects time (more detail = more work)
            length_multiplier = 1 + (description_length - 50) * 0.01

            # Energy level affects productivity (higher energy = faster completion)
            energy_multiplier = 2 - (energy_level / 10)  # Range: 1.1 to 1.9

            # Hour of day productivity factor
            if 10 <= hour_of_day <= 11 or 15 <= hour_of_day <= 16:  # Peak productivity
                time_multiplier = 0.8
            elif hour_of_day <= 9 or hour_of_day >= 17:  # Low productivity
                time_multiplier = 1.3
            else:
                time_multiplier = 1.0

            # Calculate actual time with some randomness
            actual_time = base_time * complexity_multiplier * length_multiplier * energy_multiplier * time_multiplier

            # Ensure positive time before adding noise
            actual_time = max(5, actual_time)  # Minimum 5 minutes

            # Add random variation (20% of the time)
            noise_std = actual_time * 0.2
            actual_time += np.random.normal(0, noise_std)

            # Final check to ensure still positive
            actual_time = max(5, actual_time)

            X.append([task_category, complexity, description_length, hour_of_day, energy_level])
            y.append([actual_time])

        return np.array(X), np.array(y)

    def normalize_features(self, X):
        """Normalize features for better training"""
        X_norm = X.copy().astype(float)

        # Normalize each feature to roughly 0-1 range
        X_norm[:, 0] = X_norm[:, 0] / 6.0  # task_category (0-6)
        X_norm[:, 1] = (X_norm[:, 1] - 1) / 9.0  # complexity (1-10)
        X_norm[:, 2] = (X_norm[:, 2] - 5) / 95.0  # description_length (5-100)
        X_norm[:, 3] = (X_norm[:, 3] - 8) / 11.0  # hour_of_day (8-18)
        X_norm[:, 4] = (X_norm[:, 4] - 1) / 9.0  # energy_level (1-10)

        return X_norm

    def normalize_targets(self, y):
        """Normalize target values for better training"""
        # Scale down minutes to a more reasonable range
        return y / 100.0  # Convert to "hundreds of minutes"

    def denormalize_targets(self, y_norm):
        """Convert normalized targets back to minutes"""
        return y_norm * 100.0


def demonstrate_task_time_estimator():
    """
    Demonstrate the task time estimator neural network
    """
    print("🚀 Building Neural Network Task Time Estimator from Scratch!")
    print("=" * 60)

    # Generate synthetic data
    print("\n📊 Generating synthetic task data...")
    data_generator = TaskTimeDataGenerator(user_profile="balanced")
    X, y = data_generator.generate_task_features(n_samples=1000)

    print(f"Generated {len(X)} task samples")
    print(f"Features: {X.shape[1]} (task_category, complexity, description_length, hour_of_day, energy_level)")
    print(f"Target: Actual completion time in minutes")

    # Show some sample data
    print("\n📝 Sample tasks:")
    categories = ["coding", "writing", "research", "meeting", "admin", "creative", "debugging"]
    for i in range(5):
        cat_name = categories[int(X[i, 0])]
        print(f"  {cat_name} task (complexity: {X[i, 1]}, desc_len: {X[i, 2]}, "
              f"hour: {X[i, 3]}, energy: {X[i, 4]}) → {y[i, 0]:.1f} minutes")

    # Normalize features
    X_norm = data_generator.normalize_features(X)
    y_norm = data_generator.normalize_targets(y)

    # Split data
    train_size = int(0.8 * len(X))
    indices = np.random.permutation(len(X))

    X_train = X_norm[indices[:train_size]]
    y_train = y_norm[indices[:train_size]]
    X_val = X_norm[indices[train_size:]]
    y_val = y_norm[indices[train_size:]]

    print(f"\n🔄 Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Create and train neural network
    print("\n🧠 Creating Neural Network...")
    print("Architecture: 5 → 32 → 16 → 1")

    nn = NeuralNetwork(
        input_size=5,
        hidden_layers=[32, 16],
        output_size=1,
        learning_rate=0.01
    )

    print("\n🏃 Training the network...")
    train_losses, val_losses = nn.train(
        X_train, y_train, X_val, y_val,
        epochs=1000, batch_size=64, verbose=True
    )

    # Make predictions
    print("\n🔮 Making predictions...")
    train_pred_norm = nn.predict(X_train)
    val_pred_norm = nn.predict(X_val)

    # Denormalize predictions
    train_pred = data_generator.denormalize_targets(train_pred_norm)
    val_pred = data_generator.denormalize_targets(val_pred_norm)

    # Use original y values for comparison
    y_train_orig = y[indices[:train_size]]
    y_val_orig = y[indices[train_size:]]

    # Calculate accuracy metrics
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    train_mae = mean_absolute_error(y_train_orig, train_pred)
    val_mae = mean_absolute_error(y_val_orig, val_pred)
    train_mape = mean_absolute_percentage_error(y_train_orig, train_pred)
    val_mape = mean_absolute_percentage_error(y_val_orig, val_pred)

    print(f"\n📊 Results:")
    print(f"Training MAE: {train_mae:.2f} minutes")
    print(f"Validation MAE: {val_mae:.2f} minutes")
    print(f"Training MAPE: {train_mape:.2f}%")
    print(f"Validation MAPE: {val_mape:.2f}%")

    # Show some predictions vs actual
    print(f"\n🎯 Sample Predictions vs Actual:")
    for i in range(10):
        actual = y_val_orig[i, 0]
        predicted = val_pred[i, 0]
        error = abs(actual - predicted)
        print(f"  Actual: {actual:.1f}min | Predicted: {predicted:.1f}min | Error: {error:.1f}min")

    # Plot training progress
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization

    plt.subplot(1, 2, 2)
    plt.scatter(y_val_orig, val_pred, alpha=0.6, color='blue')
    plt.plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], 'r--', lw=2)
    plt.title('Predictions vs Actual Times')
    plt.xlabel('Actual Time (minutes)')
    plt.ylabel('Predicted Time (minutes)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n✨ Success! The neural network learned to estimate task completion times!")
    print(f"   Average prediction error: {val_mae:.1f} minutes ({val_mape:.1f}%)")

    return nn, data_generator


# Run the demonstration
if __name__ == "__main__":
    model, generator = demonstrate_task_time_estimator()