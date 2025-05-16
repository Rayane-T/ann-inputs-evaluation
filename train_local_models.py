import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from NeuralNet import NeuralNet
from analyze_interesting_instances import analyze_interesting_instances

class LocalLinearModel:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        # Normaliser les données
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialiser les paramètres
        n_features = X_scaled.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Descente de gradient
        for _ in range(self.n_iterations):
            # Prédictions
            y_pred = np.dot(X_scaled, self.weights) + self.bias
            
            # Calcul des gradients
            dw = np.dot(X_scaled.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)
            
            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return np.dot(X_scaled, self.weights) + self.bias

def generate_local_dataset(X, y, instance_idx, n_samples=1000, noise_level=0.1):
    """Génère un jeu de données local autour de l'instance donnée."""
    n_features = X.shape[1]
    # Convertir en float64
    X = X.astype(np.float64)
    local_X = np.random.normal(X[instance_idx], noise_level, (n_samples, n_features))
    local_y = y[instance_idx]
    return local_X, local_y

def train_and_evaluate_local_models():
    # Charger les données
    data = pd.read_csv('iris_extended.csv')
    X = data.drop(columns=['species'])
    # Convertir les colonnes booléennes en int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    X = pd.get_dummies(X).values.astype(np.float64)
    y = pd.get_dummies(data['species']).values
    
    # Charger le modèle de réseau de neurones
    model = NeuralNet(hidden_layer_sizes=(16, 8), activation='sigmoid')
    model.load_weights('model_weights.npy.npz')
    
    # Obtenir les instances intéressantes
    interesting_instances = analyze_interesting_instances()
    
    # Sélectionner deux instances spécifiques
    selected_instances = []
    for desc, idx in interesting_instances:
        if 'Haute confiance correcte' in desc:
            selected_instances.append((desc, idx))
        elif 'Basse confiance incorrecte' in desc:
            selected_instances.append((desc, idx))
        if len(selected_instances) == 2:
            break
    
    # Pour chaque instance sélectionnée
    for desc, idx in selected_instances:
        print(f"\nAnalyse du modèle local pour l'instance {desc} (Index: {idx}):")
        print("-" * 80)
        
        # Générer le jeu de données local
        local_X, local_y = generate_local_dataset(X, y, idx)
        
        # Obtenir les prédictions du réseau de neurones
        nn_predictions = model.predict(local_X)
        
        # Entraîner un modèle local pour chaque classe
        local_models = []
        for class_idx in range(y.shape[1]):
            model_local = LocalLinearModel()
            model_local.fit(local_X, nn_predictions[:, class_idx])
            local_models.append(model_local)
        
        # Évaluer les performances du modèle local
        local_predictions = np.array([model.predict(local_X) for model in local_models]).T
        
        # Calculer l'erreur MSE entre les prédictions du réseau et du modèle local
        mse = np.mean((nn_predictions - local_predictions) ** 2)
        
        print(f"Erreur MSE du modèle local: {mse:.6f}")
        
        # Afficher les coefficients du modèle local pour la classe prédite
        pred_class = np.argmax(model.predict(X[idx:idx+1]))
        print(f"\nCoefficients du modèle local pour la classe {pred_class}:")
        for i, coef in enumerate(local_models[pred_class].weights):
            print(f"  Attribut {i}: {coef:.4f}")
        
        # Comparer les prédictions pour l'instance originale
        nn_pred = model.predict(X[idx:idx+1])[0]
        local_pred = np.array([model.predict(X[idx:idx+1])[0] for model in local_models])
        
        print("\nComparaison des prédictions pour l'instance originale:")
        print("Réseau de neurones:", nn_pred)
        print("Modèle local:", local_pred)
        print("-" * 80)

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
    train_and_evaluate_local_models() 