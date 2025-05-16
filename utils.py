import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from NeuralNet import NeuralNet

def load_data():
    """Load the iris dataset and return training and test data."""
    # Load data from CSV
    data = pd.read_csv('iris.csv')
    
    # Separate features and target
    X = data.iloc[:, :-1].values  # All columns except the last one
    y = data.iloc[:, -1].values   # Last column (target)
    
    # Convert target to numeric labels
    unique_labels = np.unique(y)
    y = np.array([np.where(unique_labels == label)[0][0] for label in y])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature names
    feature_names = data.columns[:-1].tolist()
    
    return X_train, y_train, X_test, y_test, feature_names

def load_model(model_path):
    """Load the trained neural network model."""
    # Create a new model instance with the same architecture as during training
    model = NeuralNet(
        hidden_layer_sizes=(8, 4),
        activation='sigmoid',
        batch_size=8,
        learning_rate=0.1,
        epoch=100
    )
    
    # Load the weights
    model.load_weights(model_path)
    
    return model 