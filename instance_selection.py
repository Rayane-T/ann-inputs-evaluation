import numpy as np
import pandas as pd
from NeuralNet import NeuralNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Split data with a different random state
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Convert to numpy arrays
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    
    return X_train, y_train, X_val, y_val, features

def load_model():
    # Load model with higher learning rate and fewer epochs
    nn = NeuralNet(hidden_layer_sizes=(16,8), batch_size=4, activation='tanh',
                   learning_rate=0.01, epoch=100)
    return nn

def analyze_predictions(nn, X_val, y_val, features):
    # Get predictions
    predictions = nn.predict(X_val)
    
    # Convert one-hot to class indices
    true_classes = np.argmax(y_val, axis=1)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Create DataFrame with results
    results = pd.DataFrame(X_val, columns=features)
    results['true_class'] = true_classes
    results['pred_class'] = pred_classes
    results['correct'] = true_classes == pred_classes
    
    # Add prediction probabilities
    for i in range(3):
        results[f'prob_class_{i}'] = predictions[:, i]
    
    return results

def select_interesting_instances(results):
    selected_instances = []
    
    # Select 3 correctly classified instances
    correct = results[results['correct']]
    for true_class in range(3):
        class_instances = correct[correct['true_class'] == true_class]
        if len(class_instances) > 0:
            # Select instance with highest confidence
            selected = class_instances.nlargest(1, f'prob_class_{true_class}')
            selected_instances.append(selected)
    
    # Select 3 incorrectly classified instances
    incorrect = results[~results['correct']]
    if len(incorrect) > 0:
        # Group by true class and predicted class
        grouped = incorrect.groupby(['true_class', 'pred_class'])
        
        # Select one instance from each interesting group
        for (true_class, pred_class), group in grouped:
            if len(group) > 0:
                # Select instance with highest confidence in wrong prediction
                selected = group.nlargest(1, f'prob_class_{pred_class}')
                selected_instances.append(selected)
    
    selected = pd.concat(selected_instances) if selected_instances else pd.DataFrame(columns=results.columns)
    return selected

def visualize_selected_instances(selected_instances, features):
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Selected Instances Analysis', fontsize=16)
    
    # Plot each instance
    for idx, (_, instance) in enumerate(selected_instances.iterrows()):
        row = idx // 3
        col = idx % 3
        
        # Plot feature values
        ax = axes[row, col]
        features_data = instance[features]
        sns.barplot(x=features, y=features_data, ax=ax)
        ax.set_title(f'True: {instance["true_class"]}, Pred: {instance["pred_class"]}')
        ax.tick_params(axis='x', rotation=45)
        
        # Add prediction probabilities
        for i in range(3):
            prob = instance[f'prob_class_{i}']
            ax.text(i, 0.1, f'{prob:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/selected_instances.png')
    plt.close()

def main():
    # Load data and model
    X_train, y_train, X_val, y_val, features = load_data()
    nn = load_model()
    
    # Train model if not already trained
    if not nn.has_trained:
        nn.fit(X_train, y_train, X_val, y_val)
    
    # Analyze predictions
    results = analyze_predictions(nn, X_val, y_val, features)
    
    # Select interesting instances
    selected_instances = select_interesting_instances(results)
    
    # Save selected instances
    selected_instances.to_csv('results/selected_instances.csv', index=False)
    
    # Visualize selected instances
    visualize_selected_instances(selected_instances, features)
    
    # Print summary
    print("\nCorrectly Classified Instances:")
    correct_instances = selected_instances[selected_instances['correct']]
    print(correct_instances[['true_class', 'pred_class', 'correct'] + features].to_string())
    
    print("\nIncorrectly Classified Instances:")
    incorrect_instances = selected_instances[~selected_instances['correct']]
    print(incorrect_instances[['true_class', 'pred_class', 'correct'] + features].to_string())

if __name__ == '__main__':
    main() 