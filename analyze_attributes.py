import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_attributes():
    # Charger les données
    data = pd.read_csv('iris_extended.csv')
    
    # Afficher les informations de base
    print("\nInformations sur le jeu de données:")
    print("==================================")
    print(f"Nombre d'instances: {len(data)}")
    print(f"Nombre d'attributs: {len(data.columns) - 1}")  # -1 pour la classe
    print("\nAttributs disponibles:")
    for col in data.columns[:-1]:  # Exclure la colonne de classe
        print(f"- {col}")
    
    # Statistiques descriptives
    print("\nStatistiques descriptives:")
    print("=========================")
    print(data.describe())
    
    # Corrélations entre attributs
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.iloc[:, :-1].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation des attributs')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Distribution des attributs par classe
    n_attributes = len(data.columns) - 1
    n_cols = 3
    n_rows = (n_attributes + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4*n_rows))
    for i, col in enumerate(data.columns[:-1], 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(x='class', y=col, data=data)
        plt.title(f'Distribution de {col} par classe')
    plt.tight_layout()
    plt.savefig('attribute_distributions.png')
    plt.close()
    
    # Analyse des paires d'attributs
    print("\nAnalyse des paires d'attributs les plus corrélées:")
    print("================================================")
    corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
    # Éliminer les auto-corrélations
    corr_pairs = corr_pairs[corr_pairs < 1.0]
    print(corr_pairs.head(5))
    
    return data.columns[:-1].tolist()  # Retourner la liste des attributs

if __name__ == "__main__":
    analyze_attributes() 