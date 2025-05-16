import numpy as np
import pandas as pd
from NeuralNet import NeuralNet

def analyze_interesting_instances():
    # Charger les données
    data = pd.read_csv('iris_extended.csv')
    X = data.drop(columns=['species'])
    # Encodage one-hot des colonnes non numériques
    X = pd.get_dummies(X).values
    y = pd.get_dummies(data['species']).values
    
    # Charger le modèle entraîné
    model = NeuralNet(hidden_layer_sizes=(16, 8), activation='sigmoid')
    model.load_weights('model_weights.npy.npz')
    
    # Faire les prédictions
    predictions = model.predict(X)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y, axis=1)
    
    # Trouver les instances correctement et incorrectement classées
    correct = pred_classes == true_classes
    incorrect = ~correct
    
    # Sélectionner des instances intéressantes
    interesting_instances = []
    
    # 1. Instance correctement classée avec haute confiance
    high_conf_correct = np.where(correct & (np.max(predictions, axis=1) > 0.95))[0]
    if len(high_conf_correct) > 0:
        interesting_instances.append(('Haute confiance correcte', high_conf_correct[0]))
    
    # 2. Instance correctement classée avec basse confiance
    low_conf_correct = np.where(correct & (np.max(predictions, axis=1) < 0.7))[0]
    if len(low_conf_correct) > 0:
        interesting_instances.append(('Basse confiance correcte', low_conf_correct[0]))
    
    # 3. Instance correctement classée avec confiance moyenne
    med_conf_correct = np.where(correct & (np.max(predictions, axis=1) > 0.7) & (np.max(predictions, axis=1) < 0.95))[0]
    if len(med_conf_correct) > 0:
        interesting_instances.append(('Confiance moyenne correcte', med_conf_correct[0]))
    
    # 4. Instance incorrectement classée avec haute confiance
    high_conf_incorrect = np.where(incorrect & (np.max(predictions, axis=1) > 0.9))[0]
    if len(high_conf_incorrect) > 0:
        interesting_instances.append(('Haute confiance incorrecte', high_conf_incorrect[0]))
    
    # 5. Instance incorrectement classée avec basse confiance
    low_conf_incorrect = np.where(incorrect & (np.max(predictions, axis=1) < 0.6))[0]
    if len(low_conf_incorrect) > 0:
        interesting_instances.append(('Basse confiance incorrecte', low_conf_incorrect[0]))
    
    # 6. Instance incorrectement classée avec confiance moyenne
    med_conf_incorrect = np.where(incorrect & (np.max(predictions, axis=1) > 0.6) & (np.max(predictions, axis=1) < 0.9))[0]
    if len(med_conf_incorrect) > 0:
        interesting_instances.append(('Confiance moyenne incorrecte', med_conf_incorrect[0]))
    
    # Afficher les instances intéressantes
    class_names = ['setosa', 'versicolor', 'virginica']
    print("\nInstances intéressantes sélectionnées :")
    print("-" * 80)
    
    for desc, idx in interesting_instances:
        true_class = class_names[true_classes[idx]]
        pred_class = class_names[pred_classes[idx]]
        confidence = np.max(predictions[idx])
        
        print(f"\n{desc}:")
        print(f"Index: {idx}")
        print(f"Vraie classe: {true_class}")
        print(f"Classe prédite: {pred_class}")
        print(f"Confiance: {confidence:.4f}")
        print("Attributs:")
        for attr, value in zip(data.columns[:-1], X[idx]):
            print(f"  {attr}: {value}")
        print("-" * 80)

if __name__ == '__main__':
    analyze_interesting_instances() 