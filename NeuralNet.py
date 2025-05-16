import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

import Utility

np.random.seed(42)

class NeuralNet:
    def __init__(self, hidden_layer_sizes=(16, 8), activation='sigmoid', batch_size=8,
                 learning_rate=0.01, eta_decay=True, warm_start=False, 
                 early_stopping=True, patience=10, epoch=200):
        """
        Initialise le réseau de neurones avec une architecture adaptée aux attributs étendus.
        
        Args:
            hidden_layer_sizes: Taille des couches cachées (16, 8) par défaut pour gérer plus d'attributs
            activation: Fonction d'activation ('sigmoid' par défaut)
            batch_size: Taille du batch (8 par défaut)
            learning_rate: Taux d'apprentissage (0.01 par défaut)
            eta_decay: Décroissance du taux d'apprentissage (True par défaut)
            warm_start: Réutilisation des poids existants (False par défaut)
            early_stopping: Arrêt anticipé (True par défaut)
            patience: Nombre d'époques avant arrêt anticipé (10 par défaut)
            epoch: Nombre maximum d'époques (200 par défaut)
        """
        self.n_layers = len(hidden_layer_sizes)
        self.n_inputs = 0
        self.n_outputs = 0
        self.n_epoch = epoch
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.learning_rate = learning_rate
        self.eta_decay = eta_decay
        self.warm_start = warm_start
        self.has_trained = False
        
        # Initialisation des poids et biais
        self.weights = [None] * (self.n_layers + 1)
        self.biases = [None] * (self.n_layers + 1)
        self.Z = [None] * (self.n_layers + 1)
        self.A = [None] * (self.n_layers + 1)
        self.df = [None] * (self.n_layers + 1)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        
        # Choix de la fonction d'activation
        if activation == 'tanh':
            self.activation = Utility.tanh
        elif activation == 'sigmoid':
            self.activation = Utility.sigmoid
        elif activation == 'relu':
            self.activation = Utility.relu
        elif activation == 'sintr':
            self.activation = Utility.sintr
        else:
            self.activation = Utility.identity
        
        # Historique d'entraînement
        self.train_errors = []
        self.val_errors = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Pour l'early stopping
        self.best_val_error = float('inf')
        self.best_weights = None
        self.best_biases = None
        self.patience_counter = 0

    def __weights_initialization(self, X, y):
        """Initialise les poids avec une distribution adaptée aux attributs étendus."""
        n_cols = X.shape[1]
        self.n_inputs = X.shape[1]
        
        # Initialisation des couches cachées avec Xavier/Glorot initialization
        for l in range(self.n_layers):
            n_lines = self.hidden_layer_sizes[l]
            # Utilisation de Xavier/Glorot initialization pour une meilleure convergence
            scale = np.sqrt(2.0 / (n_cols + n_lines))
            self.weights[l] = np.array(np.random.normal(0, scale, (n_lines, n_cols)), dtype=np.float64)
            self.biases[l] = np.array(np.zeros((n_lines, 1)), dtype=np.float64)
            n_cols = n_lines
        
        # Initialisation de la couche de sortie
        l_out = self.n_layers
        self.n_outputs = y.shape[1]
        scale = np.sqrt(2.0 / (n_cols + self.n_outputs))
        self.weights[l_out] = np.array(np.random.normal(0, scale, (self.n_outputs, n_cols)), dtype=np.float64)
        self.biases[l_out] = np.array(np.zeros((self.n_outputs, 1)), dtype=np.float64)

    def fit(self, X_train, y_train, X_val=None, y_val=None, val=0.2):
        """Entraîne le réseau avec gestion de l'early stopping et du learning rate decay."""
        if not self.has_trained:
            self.__weights_initialization(X_train, y_train)
        
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val, random_state=42
            )
        
        # Réinitialisation de l'historique
        self.train_errors = []
        self.val_errors = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Initialisation pour l'early stopping
        self.best_val_error = float('inf')
        self.patience_counter = 0
        
        # Barre de progression pour les époques
        epoch_pbar = tqdm(range(self.n_epoch), desc="Training Progress")
        
        for e in epoch_pbar:
            # Calcul du learning rate avec decay
            if self.eta_decay:
                current_lr = self.learning_rate / (1 + e/10)
            else:
                current_lr = self.learning_rate
            
            # Entraînement sur un batch
            X_train, y_train = shuffle(X_train, y_train)
            n_batches = X_train.shape[0] // self.batch_size
            if X_train.shape[0] % self.batch_size != 0:
                n_batches += 1
            
            batch_pbar = tqdm(range(0, X_train.shape[0], self.batch_size),
                            desc=f"Epoch {e+1}/{self.n_epoch}",
                            leave=False)
            
            ave_train_error = 0
            batch_train_error = []
            
            for start in batch_pbar:
                batch_X = X_train[start:start+self.batch_size].transpose()
                batch_y = y_train[start:start+self.batch_size].transpose()
                
                # Forward pass
                error = self.__feed_forward(batch_X, batch_y)[0]
                batch_train_error.append(error)
                ave_train_error += error
                
                # Backward pass avec le learning rate actuel
                self.__backward_pass(batch_X, batch_y, current_lr)
                
                batch_pbar.set_postfix({
                    'Train Error': f"{error:.4f}",
                    'LR': f"{current_lr:.4f}"
                })
            
            batch_pbar.close()
            
            # Calcul des erreurs et accuracies
            ave_train_error /= len(batch_train_error)
            val_error = self.__feed_forward(X_val.transpose(), y_val.transpose())[0]
            
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)
            
            train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            
            # Mise à jour de l'historique
            self.train_errors.append(ave_train_error)
            self.val_errors.append(val_error)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if self.early_stopping:
                if val_error < self.best_val_error:
                    self.best_val_error = val_error
                    self.best_weights = [w.copy() for w in self.weights]
                    self.best_biases = [b.copy() for b in self.biases]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"\nEarly stopping at epoch {e}")
                        self.weights = self.best_weights
                        self.biases = self.best_biases
                        break
            
            # Mise à jour de la barre de progression
            epoch_pbar.set_postfix({
                'Train Error': f"{ave_train_error:.4f}",
                'Val Error': f"{val_error:.4f}",
                'Train Acc': f"{train_acc:.4f}",
                'Val Acc': f"{val_acc:.4f}"
            })
            
            if e % 10 == 0:
                print(f"\n* Epoch {e} -- Error: Train: {ave_train_error:.4f} Val: {val_error:.4f}")
                print(f"  Accuracy: Train: {train_acc:.4f} Val: {val_acc:.4f}")
        
        self.has_trained = True
        self.__plot_training_history()

    def __plot_training_history(self):
        """Trace l'historique d'entraînement."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot accuracy
        ax1.plot(self.train_accuracies, label="Training Accuracy")
        ax1.plot(self.val_accuracies, label="Validation Accuracy")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy During Training')
        ax1.legend()
        ax1.grid(True)
        
        # Plot error
        ax2.plot(self.train_errors, label="Training Error")
        ax2.plot(self.val_errors, label="Validation Error")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Error')
        ax2.set_title('Model Error During Training')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def predict(self, X_batch):
        """Calcule les prédictions pour un batch d'entrées."""
        _, probabilities = self.__feed_forward(X_batch.transpose())
        return probabilities.transpose()

    def forward(self, X):
        """Alias pour predict pour la compatibilité."""
        return self.predict(X)

    def __feed_forward(self, X_batch, y_batch=None):
        """Propagation avant avec gestion des attributs étendus."""
        A = np.array(X_batch, dtype=np.float64)
        
        # Propagation à travers les couches cachées
        for l in range(self.n_layers):
            W = np.array(self.weights[l], dtype=np.float64)
            b = np.array(self.biases[l], dtype=np.float64)
            Z = np.dot(W, A) + b
            A, self.df[l] = self.activation(Z)
            self.Z[l] = Z
            self.A[l] = A
        
        # Couche de sortie
        l_out = self.n_layers
        W_out = np.array(self.weights[l_out], dtype=np.float64)
        b_out = np.array(self.biases[l_out], dtype=np.float64)
        Z = np.dot(W_out, A) + b_out
        predictions = Utility.softmax(Z)
        self.Z[l_out] = Z
        self.A[l_out] = predictions
        
        # Calcul de l'erreur si y_batch est fourni
        error = 0
        if y_batch is not None:
            error = Utility.cross_entropy_cost(predictions, y_batch)
        
        return error, predictions

    def __backward_pass(self, X_batch, y_batch, learning_rate):
        """Rétropropagation avec learning rate adaptatif."""
        delta = [None] * (self.n_layers + 1)
        dW = [None] * (self.n_layers + 1)
        db = [None] * (self.n_layers + 1)
        
        # Erreur sur la couche de sortie
        l_out = self.n_layers
        delta[l_out] = self.A[l_out] - y_batch
        
        # Rétropropagation dans les couches cachées
        for l in range(l_out-1, -1, -1):
            delta[l] = np.dot(self.weights[l+1].T, delta[l+1]) * self.df[l]
        
        # Mise à jour des paramètres
        for l in range(l_out, -1, -1):
            if l == 0:
                dW[l] = np.array(np.dot(delta[l], X_batch.T) / self.batch_size, dtype=np.float64)
            else:
                dW[l] = np.array(np.dot(delta[l], self.A[l-1].T) / self.batch_size, dtype=np.float64)
            db[l] = np.array(self.__ave_delta(delta[l]), dtype=np.float64)
            
            # Mise à jour avec le learning rate actuel
            self.weights[l] = np.array(self.weights[l], dtype=np.float64)
            self.biases[l] = np.array(self.biases[l], dtype=np.float64)
            self.weights[l] -= learning_rate * dW[l]
            self.biases[l] -= learning_rate * db[l]

    @staticmethod
    def __ave_delta(delta):
        """Calcule la moyenne des deltas pour un batch."""
        return np.array([delta.mean(axis=1)]).transpose()

    def save_weights(self, filename):
        """Sauvegarde les poids du modèle."""
        # Conversion explicite en np.array de type float64 et vérification de forme
        weights = []
        biases = []
        for l in range(self.n_layers + 1):
            w = np.array(self.weights[l], dtype=np.float64)
            b = np.array(self.biases[l], dtype=np.float64)
            # Vérification de forme
            if w.shape != (self.hidden_layer_sizes[l] if l < self.n_layers else self.n_outputs, self.n_inputs if l == 0 else self.hidden_layer_sizes[l-1]):
                raise ValueError(f"Shape mismatch in weights at layer {l}: expected {(self.hidden_layer_sizes[l] if l < self.n_layers else self.n_outputs, self.n_inputs if l == 0 else self.hidden_layer_sizes[l-1])}, got {w.shape}")
            if b.shape != (self.hidden_layer_sizes[l] if l < self.n_layers else self.n_outputs, 1):
                raise ValueError(f"Shape mismatch in biases at layer {l}: expected {(self.hidden_layer_sizes[l] if l < self.n_layers else self.n_outputs, 1)}, got {b.shape}")
            weights.append(w)
            biases.append(b)
        # Conversion en np.array avec dtype=object pour sauvegarder les listes de tableaux numpy
        weights = np.array(weights, dtype=object)
        biases = np.array(biases, dtype=object)
        np.savez(filename,
                weights=weights,
                biases=biases,
                hidden_layer_sizes=self.hidden_layer_sizes,
                n_inputs=self.n_inputs,
                n_outputs=self.n_outputs)
        print(f"Model weights saved to {filename}")

    def load_weights(self, filename):
        """Charge les poids du modèle."""
        data = np.load(filename, allow_pickle=True)
        
        self.hidden_layer_sizes = tuple(data['hidden_layer_sizes'])
        self.n_inputs = int(data['n_inputs'])
        self.n_outputs = int(data['n_outputs'])
        self.n_layers = len(self.hidden_layer_sizes)
        
        # Conversion explicite en np.array
        self.weights = [np.array(w, dtype=np.float64) for w in data['weights']]
        self.biases = [np.array(b, dtype=np.float64) for b in data['biases']]
        
        self.Z = [None] * (self.n_layers + 1)
        self.A = [None] * (self.n_layers + 1)
        self.df = [None] * (self.n_layers + 1)
        
        self.has_trained = True
        print(f"Model weights loaded from {filename}")

def main():
    """Fonction principale pour tester le réseau."""
    # Charger les données
    data = pd.read_csv('iris_extended.csv')
    
    # Encodage one-hot pour toutes les colonnes non numériques sauf la cible
    features = data.drop(columns=['species'])
    features = pd.get_dummies(features)
    X = features.values
    y = pd.get_dummies(data['species']).values
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Créer et entraîner le modèle
    model = NeuralNet(
        hidden_layer_sizes=(16, 8),
        activation='sigmoid',
        batch_size=8,
        learning_rate=0.01,
        eta_decay=True,
        early_stopping=True,
        patience=10,
        epoch=200
    )
    
    model.fit(X_train, y_train, X_val, y_val)
    
    # Sauvegarder les poids
    model.save_weights('model_weights.npy')

if __name__ == '__main__':
    main()