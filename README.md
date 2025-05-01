# Évaluation de la contribution des attributs d'entrée dans un réseau de neurones

## Installation
```bash
pip install -r requirements.txt
```

## Structure du projet
```
.
├── NeuralNet.py          # Implémentation du réseau de neurones
├── Utility.py           # Fonctions utilitaires
├── iris.csv             # Dataset Iris
├── requirements.txt     # Dépendances
└── notebooks/           # Notebooks pour chaque partie
    ├── 1_neural_net.ipynb
    ├── 2_instance_selection.ipynb
    ├── 3_perturbed_instances.ipynb
    ├── 4_local_models.ipynb
    ├── 5_contribution_analysis.ipynb
    └── 6_conclusion.ipynb
```

## Tests indépendants

### 1. Réseau de neurones
```bash
python NeuralNet.py
```
Ce script entraîne le réseau de neurones avec :
- 2 couches cachées (16, 8 neurones)
- Activation tanh
- Mini-batch size = 4
- Learning rate = 0.01
- 100 époques

### 2. Sélection d'instances
```bash
python instance_selection.py
```
Ce script :
- Charge le modèle entraîné
- Analyse les prédictions sur l'ensemble de validation
- Sélectionne 3 instances correctement et 3 incorrectement classées
- Sauvegarde les instances sélectionnées

### 3. Génération d'instances perturbées
```bash
python generate_perturbed.py
```
Ce script :
- Charge les instances sélectionnées
- Génère 250 versions perturbées pour chaque instance
- Ajoute un bruit de ±10%
- Sauvegarde les instances perturbées

### 4. Modèles locaux
```bash
python train_local_models.py
```
Ce script :
- Charge les instances perturbées
- Entraîne un modèle local pour chaque instance
- Sauvegarde les modèles locaux

### 5. Analyse des contributions
```bash
python analyze_contributions.py
```
Ce script :
- Charge les modèles locaux
- Calcule les contributions des attributs
- Génère les visualisations

### 6. Conclusion
```bash
python conclusion.py
```
Ce script :
- Analyse la robustesse de l'approche
- Compare les résultats
- Génère le rapport final

## Visualisation des résultats
Chaque script génère des visualisations dans le dossier `results/` :
- Courbes d'apprentissage
- Matrices de confusion
- Graphiques de contribution
- Comparaisons des prédictions

## Dépendances
- numpy
- scikit-learn
- matplotlib
- pandas
- tqdm
- jupyter (pour les notebooks)