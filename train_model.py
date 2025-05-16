import numpy as np
from NeuralNet import NeuralNet
from utils import load_data

def main():
    # Load data
    X_train, y_train, X_test, y_test, feature_names = load_data()
    
    # Convert labels to one-hot encoding
    n_classes = len(np.unique(y_train))
    y_train_onehot = np.zeros((len(y_train), n_classes))
    y_train_onehot[np.arange(len(y_train)), y_train] = 1
    
    # Create and train model
    model = NeuralNet(
        hidden_layer_sizes=(8, 4),  # Two hidden layers
        activation='sigmoid',
        batch_size=8,
        learning_rate=0.1,
        epoch=100
    )
    
    # Train the model
    model.fit(X_train, y_train_onehot)
    
    # Save the model weights
    model.save_weights('model_weights.npy')
    print("Model trained and weights saved to 'model_weights.npy'")

if __name__ == "__main__":
    main() 