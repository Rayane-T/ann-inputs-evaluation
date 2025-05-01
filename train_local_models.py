import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and store loss
            loss = mean_squared_error(y, y_pred)
            self.loss_history.append(loss)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def load_perturbed_data(instance_id):
    """Load perturbed instances and their predictions for a given instance."""
    perturbed_path = f'results/perturbed_instances/instance_{instance_id}_perturbed.csv'
    predictions_path = f'results/perturbed_instances/instance_{instance_id}_predictions.csv'
    
    X = pd.read_csv(perturbed_path).values
    y = pd.read_csv(predictions_path).values
    
    return X, y

def train_local_models(instance_id, X, y, features):
    """Train local models for each class probability."""
    models = []
    predictions = []
    
    # Train a model for each class probability
    for class_idx in range(3):
        # Create and train model
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(X, y[:, class_idx])
        models.append(model)
        
        # Make predictions
        y_pred = model.predict(X)
        predictions.append(y_pred)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(features, model.weights)
        plt.title(f'Feature Importance for Class {class_idx}')
        plt.xlabel('Features')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/local_models/instance_{instance_id}_class_{class_idx}_importance.png')
        plt.close()
        
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_history)
        plt.title(f'Loss History for Class {class_idx}')
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.tight_layout()
        plt.savefig(f'results/local_models/instance_{instance_id}_class_{class_idx}_loss.png')
        plt.close()
    
    return models, np.array(predictions).T

def visualize_predictions(instance_id, y_true, y_pred):
    """Visualize true vs predicted probabilities."""
    plt.figure(figsize=(10, 6))
    for class_idx in range(3):
        plt.scatter(y_true[:, class_idx], y_pred[:, class_idx], 
                   label=f'Class {class_idx}', alpha=0.5)
    
    # Add perfect prediction line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Prediction')
    
    plt.title('True vs Predicted Probabilities')
    plt.xlabel('True Probability')
    plt.ylabel('Predicted Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/local_models/instance_{instance_id}_predictions.png')
    plt.close()

def visualize_contributions_comparison(instance_id, models, features, true_class, predicted_class, correct):
    """Create a comparative visualization of feature contributions."""
    plt.figure(figsize=(15, 5))
    
    # Plot contributions for each class
    for class_idx, model in enumerate(models):
        plt.subplot(1, 3, class_idx + 1)
        weights = np.abs(model.weights)  # Use absolute values for contribution
        plt.bar(features, weights)
        plt.title(f'Class {class_idx} Contributions')
        plt.xlabel('Features')
        plt.ylabel('Absolute Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.suptitle(f'Instance {instance_id} - True: {true_class}, Predicted: {predicted_class}, Correct: {correct}')
    plt.tight_layout()
    plt.savefig(f'results/local_models/instance_{instance_id}_contributions_comparison.png')
    plt.close()

def main():
    # Create directory for results
    os.makedirs('results/local_models', exist_ok=True)
    
    # Load selected instances
    selected_instances = pd.read_csv('results/selected_instances.csv')
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Process each instance
    for idx, (_, instance) in enumerate(selected_instances.iterrows()):
        print(f"\nProcessing instance {idx}:")
        print(f"True class: {instance['true_class']}")
        print(f"Predicted class: {instance['pred_class']}")
        print(f"Correct: {instance['correct']}")
        
        # Load perturbed data
        X, y = load_perturbed_data(idx)
        
        # Train local models
        models, y_pred = train_local_models(idx, X, y, features)
        
        # Visualize predictions
        visualize_predictions(idx, y, y_pred)
        
        # Visualize contributions comparison
        visualize_contributions_comparison(
            idx, models, features, 
            instance['true_class'], 
            instance['pred_class'], 
            instance['correct']
        )
        
        # Print model summary
        print("\nModel Summary:")
        for class_idx, model in enumerate(models):
            print(f"\nClass {class_idx}:")
            print("Feature Contributions (absolute weights):")
            for feature, weight in zip(features, np.abs(model.weights)):
                print(f"  {feature}: {weight:.4f}")
            print(f"Bias: {model.bias:.4f}")
            print(f"Final Loss: {model.loss_history[-1]:.4f}")

if __name__ == '__main__':
    main() 