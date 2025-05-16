import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from NeuralNet import NeuralNet
from utils import load_data, load_model

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def select_interesting_instances(X_test, y_test, model, num_instances=4):
    """Select interesting instances based on prediction confidence and correctness."""
    # Get predictions
    predictions = model.forward(X_test)
    probabilities = softmax(predictions)
    max_probs = np.max(probabilities, axis=1)
    predicted_classes = np.argmax(probabilities, axis=1)
    
    # Calculate prediction confidence and correctness
    confidence = max_probs
    correct = (predicted_classes == y_test)
    
    # Select instances with interesting characteristics
    interesting_indices = []
    
    # 1. Correctly classified with high confidence
    high_conf_correct = np.where((confidence > 0.9) & correct)[0]
    if len(high_conf_correct) > 0:
        interesting_indices.append(high_conf_correct[0])
        
    # 2. Correctly classified with low confidence
    low_conf_correct = np.where((confidence < 0.7) & correct)[0]
    if len(low_conf_correct) > 0:
        interesting_indices.append(low_conf_correct[0])
        
    # 3. Incorrectly classified with high confidence
    high_conf_incorrect = np.where((confidence > 0.9) & ~correct)[0]
    if len(high_conf_incorrect) > 0:
        interesting_indices.append(high_conf_incorrect[0])
        
    # 4. Incorrectly classified with low confidence
    low_conf_incorrect = np.where((confidence < 0.7) & ~correct)[0]
    if len(low_conf_incorrect) > 0:
        interesting_indices.append(low_conf_incorrect[0])
        
    return interesting_indices[:num_instances]

def build_local_model(X_train, y_train, X_test, instance_idx, max_depth=4, model=None):
    """Build a local decision tree model for a specific instance."""
    # Create a local dataset by weighting training instances based on distance
    instance = X_test[instance_idx]
    distances = np.linalg.norm(X_train - instance, axis=1)
    weights = np.exp(-distances / np.mean(distances))
    
    # Use the neural network to get soft labels (probabilities)
    nn_probs = model.forward(X_train)
    
    # Train decision tree to predict the neural network's output (probabilities)
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X_train, nn_probs, sample_weight=weights)
    
    return tree

def evaluate_local_model(tree, X_test, y_test, instance_idx, model):
    """Evaluate how well the local model approximates the neural network."""
    # Get neural network predictions
    nn_predictions = model.forward(X_test)
    nn_probs = softmax(nn_predictions)
    
    # Get local model predictions
    local_predictions = tree.predict(X_test)
    
    # Calculate MSE between local and neural network predictions
    mse = mean_squared_error(nn_probs, local_predictions)
    
    return mse

def plot_feature_importance(tree, feature_names):
    """Plot feature importance from the local model."""
    importances = tree.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance in Local Model')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # Load data and model
    X_train, y_train, X_test, y_test, feature_names = load_data()
    model = load_model('model_weights.npy.npz')
    
    # Select interesting instances
    interesting_indices = select_interesting_instances(X_test, y_test, model)
    
    print("\nSelected Interesting Instances:")
    print("===============================")
    for idx in interesting_indices:
        prediction = model.forward(X_test[idx:idx+1])
        prob = softmax(prediction)
        max_prob = np.max(prob)
        pred_class = np.argmax(prob)
        correct = pred_class == y_test[idx]
        
        print(f"\nInstance {idx}:")
        print(f"True class: {y_test[idx]}")
        print(f"Predicted class: {pred_class}")
        print(f"Confidence: {max_prob:.3f}")
        print(f"Correctly classified: {correct}")
    
    # Build and evaluate local model for the first interesting instance
    instance_idx = interesting_indices[0]
    local_model = build_local_model(X_train, y_train, X_test, instance_idx, model=model)
    
    # Evaluate local model
    mse = evaluate_local_model(local_model, X_test, y_test, instance_idx, model)
    print(f"\nLocal Model Performance (MSE): {mse:.4f}")
    
    # Plot feature importance
    plot_feature_importance(local_model, feature_names)
    print("\nFeature importance plot saved as 'feature_importance.png'")

if __name__ == "__main__":
    main() 