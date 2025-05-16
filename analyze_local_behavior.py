import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NeuralNet import NeuralNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to numpy arrays
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    
    return X_train, y_train, X_val, y_val, features

def select_interesting_instances(nn, X_val, y_val, features):
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
    
    selected_instances = []
    reasons = []
    
    # 1. Instance avec haute confiance et classification correcte
    correct_high_conf = results[results['correct']].nlargest(1, f'prob_class_{results["true_class"].iloc[0]}')
    if not correct_high_conf.empty:
        selected_instances.append(correct_high_conf.iloc[0])
        reasons.append("Instance avec haute confiance et classification correcte")
    
    # 2. Instance avec classification incorrecte mais haute confiance
    incorrect_high_conf = results[~results['correct']].nlargest(1, f'prob_class_{results["pred_class"].iloc[0]}')
    if not incorrect_high_conf.empty:
        selected_instances.append(incorrect_high_conf.iloc[0])
        reasons.append("Instance avec classification incorrecte mais haute confiance")
    
    # 3. Instance avec probabilités équilibrées
    balanced = results.iloc[np.argmin(np.max(predictions, axis=1))]
    selected_instances.append(balanced)
    reasons.append("Instance avec probabilités équilibrées entre les classes")
    
    return pd.DataFrame(selected_instances), reasons

def generate_perturbed_instances(instance, n_samples=100):
    """Generate perturbed instances around the selected instance."""
    # S'assurer que l'instance est un vecteur numpy 1D
    instance = np.asarray(instance).flatten()
    perturbed = []
    for _ in range(n_samples):
        # Add small random perturbations
        noise = np.random.normal(0, 0.1, size=instance.shape)
        perturbed.append(instance + noise)
    return np.array(perturbed)

def train_local_model(X, y):
    """Train a simple linear regression model."""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    return model

def analyze_local_behavior(instance, nn, features):
    """Analyze the local behavior of the neural network around an instance."""
    # Generate perturbed instances
    X_perturbed = generate_perturbed_instances(instance[features].values)
    
    # Get neural network predictions
    y_nn = nn.predict(X_perturbed)
    
    # Train local models for each class
    local_models = []
    r2_scores = []
    
    for class_idx in range(3):
        model = train_local_model(X_perturbed, y_nn[:, class_idx])
        local_models.append(model)
        
        # Calculate R² score
        y_pred = model.predict(X_perturbed)
        r2 = r2_score(y_nn[:, class_idx], y_pred)
        r2_scores.append(r2)
    
    return local_models, r2_scores

def visualize_local_behavior(instance, nn, local_models, features, reason):
    """Visualize the local behavior of the neural network and local models."""
    # Generate test points
    X_test = generate_perturbed_instances(instance[features].values, n_samples=50)
    y_nn = nn.predict(X_test)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Analyse du comportement local\nRaison de sélection: {reason}', fontsize=14)
    
    # Plot 1: Feature importance
    ax = axes[0, 0]
    importance = np.abs(local_models[0].coef_)
    sns.barplot(x=features, y=importance, ax=ax)
    ax.set_title('Importance des caractéristiques')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: True vs Predicted for each class
    ax = axes[0, 1]
    for class_idx in range(3):
        y_pred = local_models[class_idx].predict(X_test)
        ax.scatter(y_nn[:, class_idx], y_pred, 
                  label=f'Classe {class_idx}', alpha=0.5)
    ax.plot([0, 1], [0, 1], 'k--', label='Prédiction parfaite')
    ax.set_title('Vraies vs Prédites probabilités')
    ax.set_xlabel('Probabilité vraie')
    ax.set_ylabel('Probabilité prédite')
    ax.legend()
    
    # Plot 3: Feature contributions
    ax = axes[1, 0]
    contributions = np.array([model.coef_ for model in local_models])
    sns.heatmap(contributions, annot=True, fmt='.2f', 
                xticklabels=features, yticklabels=[f'Classe {i}' for i in range(3)],
                ax=ax, cmap='RdBu', center=0)
    ax.set_title('Contributions des caractéristiques par classe')
    
    # Plot 4: R² scores
    ax = axes[1, 1]
    r2_scores = [r2_score(y_nn[:, i], local_models[i].predict(X_test)) 
                 for i in range(3)]
    sns.barplot(x=[f'Classe {i}' for i in range(3)], y=r2_scores, ax=ax)
    ax.set_title('Scores R² par classe')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/local_behavior_analysis.png')
    plt.close()

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load data and model
    X_train, y_train, X_val, y_val, features = load_data()
    nn = NeuralNet(hidden_layer_sizes=(16,8), batch_size=4, activation='tanh',
                   learning_rate=0.01, epoch=100)
    
    # Train model if not already trained
    if not nn.has_trained:
        nn.fit(X_train, y_train, X_val, y_val)
    
    # Select interesting instances
    selected_instances, reasons = select_interesting_instances(nn, X_val, y_val, features)
    
    # Save selected instances
    selected_instances.to_csv('results/selected_instances.csv', index=False)
    
    # Print selected instances and reasons
    print("\nInstances sélectionnées et raisons:")
    for idx, (_, instance) in enumerate(selected_instances.iterrows()):
        print(f"\nInstance {idx + 1}:")
        print(f"Raison: {reasons[idx]}")
        print(f"Vraie classe: {instance['true_class']}")
        print(f"Classe prédite: {instance['pred_class']}")
        print(f"Correct: {instance['correct']}")
        print("\nCaractéristiques:")
        for feature in features:
            print(f"  {feature}: {instance[feature]:.2f}")
        print("\nProbabilités prédites:")
        for i in range(3):
            print(f"  Classe {i}: {instance[f'prob_class_{i}']:.4f}")
    
    # Analyze local behavior for the first instance
    instance = selected_instances.iloc[0]
    local_models, r2_scores = analyze_local_behavior(instance, nn, features)
    
    # Visualize local behavior
    visualize_local_behavior(instance, nn, local_models, features, reasons[0])
    
    # Print local model analysis
    print("\nAnalyse du modèle local:")
    print(f"Raison de sélection: {reasons[0]}")
    print("\nScores R² par classe:")
    for i, r2 in enumerate(r2_scores):
        print(f"  Classe {i}: {r2:.4f}")
    
    print("\nContributions des caractéristiques:")
    for i, model in enumerate(local_models):
        print(f"\nClasse {i}:")
        for feature, coef in zip(features, model.coef_):
            print(f"  {feature}: {coef:.4f}")

if __name__ == '__main__':
    main() 