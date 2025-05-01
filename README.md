# ANN Inputs Evaluation

Ce projet explore l'évaluation des entrées d'un réseau de neurones artificiels (ANN) sur le jeu de données Iris. Il comprend plusieurs scripts pour analyser les prédictions, générer des instances perturbées, et évaluer différentes architectures de réseaux de neurones.

## Structure du Projet

```
.
├── NeuralNet.py              # Implémentation du réseau de neurones
├── analyze_predictions.py    # Analyse des prédictions du modèle
├── confusion_matrix.py       # Analyse des performances avec différentes architectures
├── generate_perturbed_instances.py  # Génération d'instances perturbées
├── train_local_models.py     # Entraînement de modèles locaux interprétables
└── results/                  # Dossier contenant les résultats
    ├── distributions/        # Résultats des analyses de distributions
    │   ├── data/            # Données brutes (CSV)
    │   └── plots/           # Visualisations (PNG)
    └── perturbed_instances/  # Résultats des instances perturbées
```

## Scripts

### 1. analyze_predictions.py
Analyse les prédictions du modèle sur le jeu de données Iris :
- Charge et prépare les données
- Entraîne le modèle
- Calcule et visualise les distributions de prédictions
- Sauvegarde les résultats dans `results/distributions/`

### 2. confusion_matrix.py
Évalue différentes architectures de réseaux de neurones :
- Teste 6 configurations différentes :
  - (3,) : Une couche de 3 neurones
  - (3, 2) : Deux couches (3 et 2 neurones)
  - (8,) : Une couche de 8 neurones
  - (8, 4) : Deux couches (8 et 4 neurones)
  - (16, 8) : Deux couches (16 et 8 neurones)
  - (16, 8, 4) : Trois couches (16, 8 et 4 neurones)
- Génère des matrices de confusion
- Calcule les métriques de performance (précision, rappel, F1-score)
- Sauvegarde les visualisations dans `results/distributions/plots/`

### 3. generate_perturbed_instances.py
Génère des versions perturbées des instances du jeu de données :
- Sélectionne des instances du jeu de test
- Applique des perturbations aléatoires (±10%)
- Génère 250 versions perturbées par instance
- Sauvegarde les résultats dans `results/perturbed_instances/`

### 4. train_local_models.py
Entraîne des modèles locaux interprétables :
- Implémente un modèle de régression linéaire
- Entraîne des modèles locaux pour chaque classe
- Visualise l'importance des caractéristiques
- Sauvegarde les résultats dans `results/local_models/`

## Résultats

Les résultats sont organisés dans le dossier `results/` :

- `distributions/` : Contient les analyses des distributions de prédictions
  - `data/` : Fichiers CSV avec les données brutes
  - `plots/` : Visualisations des résultats
- `perturbed_instances/` : Contient les instances perturbées et leurs prédictions
- `local_models/` : Contient les résultats des modèles locaux

## Utilisation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Exécuter les scripts dans l'ordre :
```bash
python3 analyze_predictions.py
python3 confusion_matrix.py
python3 generate_perturbed_instances.py
python3 train_local_models.py
```

## Dépendances

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn