import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from NeuralNet import NeuralNet
import os

def load_data():
    """Load and prepare the Iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # One-hot encode the target
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names

def load_model():
    """Load or create the neural network model."""
    model = NeuralNet(
        hidden_layer_sizes=(16, 8),
        activation='tanh',
        batch_size=32,
        learning_rate=0.01,
        epoch=100
    )
    return model

def analyze_predictions(model, X_test, y_test, feature_names, target_names):
    """Analyze predictions for a batch of 4 instances."""
    # Create subdirectories
    os.makedirs('results/distributions/plots', exist_ok=True)
    os.makedirs('results/distributions/data', exist_ok=True)
    
    # Select 4 random instances
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(len(X_test), 4, replace=False)
    X_batch = X_test[indices]
    y_batch = y_test[indices]
    
    # Get predictions
    predictions = model.predict(X_batch)
    
    # Create DataFrame for distributions
    distributions_data = []
    for i in range(len(X_batch)):
        instance_data = {
            'Instance': i + 1,
            'True_Class': target_names[np.argmax(y_batch[i])],
            'Predicted_Class': target_names[np.argmax(predictions[i])]
        }
        # Add feature values
        for name, value in zip(feature_names, X_batch[i]):
            instance_data[f'Feature_{name}'] = value
        # Add true probabilities
        for j, name in enumerate(target_names):
            instance_data[f'True_Prob_{name}'] = y_batch[i][j]
        # Add predicted probabilities
        for j, name in enumerate(target_names):
            instance_data[f'Pred_Prob_{name}'] = predictions[i][j]
        
        distributions_data.append(instance_data)
    
    # Save distributions to CSV
    df = pd.DataFrame(distributions_data)
    df.to_csv('results/distributions/data/distributions.csv', index=False)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot for each instance
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        # Convert one-hot to class index
        true_class = np.argmax(y_batch[i])
        pred_class = np.argmax(predictions[i])
        
        # Create bar plot
        x = np.arange(len(target_names))
        width = 0.35
        
        ax.bar(x - width/2, y_batch[i], width, label='True Probabilities')
        ax.bar(x + width/2, predictions[i], width, label='Predicted Probabilities')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Probability')
        ax.set_title(f'Instance {i+1}\nTrue: {target_names[true_class]}, Predicted: {target_names[pred_class]}')
        ax.set_xticks(x)
        ax.set_xticklabels(target_names)
        ax.legend()
        
        # Add feature values
        feature_text = '\n'.join([f'{name}: {value:.2f}' for name, value in zip(feature_names, X_batch[i])])
        ax.text(0.02, 0.98, feature_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/distributions/plots/prediction_distributions.png')
    plt.close()
    
    # Print detailed information
    print("\nDetailed Analysis of Predictions:")
    for i, idx in enumerate(indices):
        true_class = np.argmax(y_batch[i])
        pred_class = np.argmax(predictions[i])
        print(f"\nInstance {i+1}:")
        print(f"True Class: {target_names[true_class]}")
        print(f"Predicted Class: {target_names[pred_class]}")
        print("Feature Values:")
        for name, value in zip(feature_names, X_batch[i]):
            print(f"  {name}: {value:.2f}")
        print("\nClass Probabilities:")
        for j, (true_prob, pred_prob) in enumerate(zip(y_batch[i], predictions[i])):
            print(f"  {target_names[j]}: True={true_prob:.4f}, Predicted={pred_prob:.4f}")

def main():
    # Create results directory
    os.makedirs('results/distributions/plots', exist_ok=True)
    os.makedirs('results/distributions/data', exist_ok=True)
    
    # Load data and model
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    model = load_model()
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train, X_test, y_test)
    model.save_weights('results/distributions/data/model_weights.npz')
    
    # Analyze predictions
    analyze_predictions(model, X_test, y_test, feature_names, target_names)

if __name__ == '__main__':
    main() 