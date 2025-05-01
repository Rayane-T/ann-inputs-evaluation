import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
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
    
    return X_train, X_test, y_train, y_test, iris.target_names

def train_and_evaluate_model(hidden_sizes, activation='tanh', batch_size=32, learning_rate=0.01, epoch=100):
    """Train and evaluate a model with given configuration."""
    model = NeuralNet(
        hidden_layer_sizes=hidden_sizes,
        activation=activation,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epoch=epoch
    )
    
    # Load data
    X_train, X_test, y_train, y_test, class_names = load_data()
    
    # Train model
    print(f"\nTraining model with hidden layers: {hidden_sizes}")
    model.fit(X_train, y_train, X_test, y_test)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Convert one-hot to class indices
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average=None)
    recall = recall_score(y_true_classes, y_pred_classes, average=None)
    f1 = f1_score(y_true_classes, y_pred_classes, average=None)
    
    return {
        'hidden_sizes': hidden_sizes,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes)
    }

def analyze_results(results, class_names):
    """Analyze and visualize results from different model configurations."""
    # Create results directory
    os.makedirs('results/distributions/plots', exist_ok=True)
    os.makedirs('results/distributions/data', exist_ok=True)
    
    # Prepare data for visualization
    configs = [str(r['hidden_sizes']) for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.bar(configs, accuracies)
    plt.title('Model Accuracy by Configuration')
    plt.xlabel('Hidden Layer Sizes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/distributions/plots/accuracy_comparison.png')
    plt.close()
    
    # Plot F1-scores for each class
    plt.figure(figsize=(15, 8))
    for i, class_name in enumerate(class_names):
        f1_scores = [r['f1'][i] for r in results]
        plt.plot(configs, f1_scores, marker='o', label=class_name)
    
    plt.title('F1-Scores by Class and Configuration')
    plt.xlabel('Hidden Layer Sizes')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/distributions/plots/f1_comparison.png')
    plt.close()
    
    # Plot confusion matrix for (8,) configuration
    best_config_idx = np.argmax(accuracies)
    best_config = results[best_config_idx]
    cm_df = pd.DataFrame(best_config['confusion_matrix'], 
                        index=class_names,
                        columns=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for Configuration {best_config["hidden_sizes"]}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('results/distributions/plots/best_confusion_matrix.png')
    plt.close()
    
    # Print detailed analysis
    print("\nDetailed Analysis of Different Configurations:")
    for result in results:
        print(f"\nConfiguration: {result['hidden_sizes']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("\nF1-Scores by class:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {result['f1'][i]:.4f}")
        
        # Identify most difficult class
        most_difficult = class_names[np.argmin(result['f1'])]
        print(f"Most difficult class: {most_difficult}")
        print(f"F1-Score: {result['f1'][np.argmin(result['f1'])]:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'hidden_sizes': str(r['hidden_sizes']),
            'accuracy': r['accuracy'],
            'setosa_f1': r['f1'][0],
            'versicolor_f1': r['f1'][1],
            'virginica_f1': r['f1'][2]
        } for r in results
    ])
    results_df.to_csv('results/distributions/data/model_comparison.csv', index=False)

def main():
    # Define different network configurations to test
    configurations = [
        (3,),          # Simple network
        (3, 2),        # Two small layers
        (8,),          # Single larger layer
        (8, 4),        # Two medium layers
        (16, 8),       # Two larger layers
        (16, 8, 4),    # Three layers
    ]
    
    # Train and evaluate each configuration
    results = []
    for config in configurations:
        result = train_and_evaluate_model(config)
        results.append(result)
    
    # Analyze results
    _, _, _, _, class_names = load_data()
    analyze_results(results, class_names)

if __name__ == '__main__':
    main() 