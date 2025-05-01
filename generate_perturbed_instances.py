import numpy as np
import pandas as pd
from NeuralNet import NeuralNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    # Load data
    iris_df = pd.read_csv('iris.csv')
    
    # Extract features and target
    df_columns = iris_df.columns.values.tolist()
    features = df_columns[0:4]
    label = df_columns[4:] # ['class']
    
    X = iris_df[features]
    y = iris_df[label]
    y = pd.get_dummies(y, dtype='int') # one-hot encoding
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Convert to numpy arrays
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    
    return X_train, y_train, X_val, y_val, features

def load_model():
    # Load trained model
    nn = NeuralNet(hidden_layer_sizes=(16,8), batch_size=4, activation='tanh',
                   learning_rate=0.01, epoch=100)
    return nn

def generate_perturbed_instances(instance, n_perturbations=250, noise_level=0.1):
    """
    Generate perturbed versions of an instance by adding random noise.
    
    Args:
        instance: Original instance (numpy array)
        n_perturbations: Number of perturbed instances to generate
        noise_level: Maximum noise level as a fraction of the original value (±10%)
    
    Returns:
        perturbed_instances: Array of perturbed instances
    """
    # Ensure instance is a numpy array
    instance = np.array(instance, dtype=np.float64)
    
    perturbed_instances = []
    
    for _ in range(n_perturbations):
        # Generate random noise between -noise_level and +noise_level
        noise = np.random.uniform(-noise_level, noise_level, size=instance.shape)
        # Apply noise to original instance
        perturbed = instance * (1 + noise)
        perturbed_instances.append(perturbed)
    
    return np.array(perturbed_instances, dtype=np.float64)

def predict_perturbed_instances(nn, perturbed_instances):
    """
    Get predictions for perturbed instances.
    
    Args:
        nn: Trained neural network
        perturbed_instances: Array of perturbed instances
    
    Returns:
        predictions: Array of predictions (class probabilities)
    """
    return nn.predict(perturbed_instances)

def save_perturbed_data(instance, perturbed_instances, predictions, features, instance_id):
    """
    Save perturbed instances and their predictions to CSV files.
    
    Args:
        instance: Original instance
        perturbed_instances: Array of perturbed instances
        predictions: Array of predictions
        features: List of feature names
        instance_id: ID of the original instance
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/perturbed_instances', exist_ok=True)
    
    # Save perturbed instances
    perturbed_df = pd.DataFrame(perturbed_instances, columns=features)
    perturbed_df.to_csv(f'results/perturbed_instances/instance_{instance_id}_perturbed.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions, columns=[f'prob_class_{i}' for i in range(3)])
    predictions_df.to_csv(f'results/perturbed_instances/instance_{instance_id}_predictions.csv', index=False)
    
    # Save original instance
    original_df = pd.DataFrame([instance], columns=features)
    original_df.to_csv(f'results/perturbed_instances/instance_{instance_id}_original.csv', index=False)

def visualize_perturbations(instance, perturbed_instances, predictions, features, instance_id):
    """
    Create visualizations of the perturbations and their effects.
    
    Args:
        instance: Original instance
        perturbed_instances: Array of perturbed instances
        predictions: Array of predictions
        features: List of feature names
        instance_id: ID of the original instance
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Perturbation Analysis for Instance {instance_id}', fontsize=16)
    
    # Plot feature distributions
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Plot original value
        ax.axvline(x=instance[i], color='r', linestyle='--', label='Original')
        
        # Plot perturbed values
        sns.histplot(perturbed_instances[:, i], ax=ax, bins=20)
        
        ax.set_title(f'{feature} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/perturbed_instances/instance_{instance_id}_distributions.png')
    plt.close()
    
    # Plot prediction distributions
    plt.figure(figsize=(10, 6))
    for i in range(3):
        sns.histplot(predictions[:, i], label=f'Class {i}', bins=20)
    plt.title('Prediction Probability Distributions')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'results/perturbed_instances/instance_{instance_id}_predictions.png')
    plt.close()

def main():
    # Load data and model
    X_train, y_train, X_val, y_val, features = load_data()
    nn = load_model()
    
    # Train model if not already trained
    if not nn.has_trained:
        nn.fit(X_train, y_train, X_val, y_val)
    
    # Load selected instances
    selected_instances = pd.read_csv('results/selected_instances.csv')
    
    # Process each selected instance
    for idx, (_, instance) in enumerate(selected_instances.iterrows()):
        # Extract features
        instance_features = instance[features].values
        
        # Generate perturbed instances
        perturbed_instances = generate_perturbed_instances(instance_features)
        
        # Get predictions for perturbed instances
        predictions = predict_perturbed_instances(nn, perturbed_instances)
        
        # Save data
        save_perturbed_data(instance_features, perturbed_instances, predictions, features, idx)
        
        # Create visualizations
        visualize_perturbations(instance_features, perturbed_instances, predictions, features, idx)
        
        print(f"\nProcessed instance {idx}:")
        print(f"Original class: {instance['true_class']}")
        print(f"Predicted class: {instance['pred_class']}")
        print(f"Correct: {instance['correct']}")

if __name__ == '__main__':
    main() 