import numpy as np
import pandas as pd
from NeuralNet import NeuralNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
iris_df = pd.read_csv('iris.csv')
features = iris_df.columns[:4].tolist()
label = iris_df.columns[4]

X = iris_df[features]
y = pd.get_dummies(iris_df[label], dtype='int')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_val, y_val = X_val.to_numpy(), y_val.to_numpy()

# Charger le modèle
nn = NeuralNet(hidden_layer_sizes=(16, 8), batch_size=4, activation='tanh', learning_rate=0.01, epoch=100)
nn.fit(X_train, y_train)

# Sélectionner une instance bien classée
preds = nn.predict(X_val)
original_classes = np.argmax(y_val, axis=1)
predicted_classes = np.argmax(preds, axis=1)

correct_indices = np.where(original_classes == predicted_classes)[0]
instance_index = correct_indices[0]
original_instance = X_val[instance_index]
original_class = predicted_classes[instance_index]

# Générer 250 perturbations
def generate_perturbations(instance, n=250, noise_level=0.1):
    perturbations = []
    for _ in range(n):
        noise = np.random.uniform(-noise_level, noise_level, size=instance.shape)
        perturbed = instance * (1 + noise)
        perturbations.append(perturbed)
    return np.array(perturbations)

perturbed_instances = generate_perturbations(original_instance)
perturbed_preds = nn.predict(perturbed_instances)
perturbed_classes = np.argmax(perturbed_preds, axis=1)

# Analyse des prédictions
same = np.sum(perturbed_classes == original_class)
diff = len(perturbed_classes) - same

# Affichage texte
print(f"Sur 250 instances perturbées : {same} ont gardé la même classe, {diff} ont changé de prédiction.")

# Graphique
plt.figure(figsize=(6, 6))
plt.pie([same, diff], labels=["Même prédiction", "Prédiction différente"], autopct="%1.1f%%", startangle=90)
plt.title("Robustesse locale autour d'une instance test")
plt.show()
