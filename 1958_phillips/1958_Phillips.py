import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement des fichiers de données
# Assurez-vous que les fichiers CSV sont dans le même répertoire que votre script
df_chomage = pd.read_csv('DATA/Unemployment Rate in the UK_1760-2016_annuel.csv')
df_inflation = pd.read_csv('DATA/Consumer Price Inflation in the UK 1210-2016.csv')

# 2. Préparation des données : extraction de l'année
# On extrait l'année pour pouvoir fusionner les deux bases de données
df_chomage['year'] = df_chomage['observation_date'].str[:4].astype(int)
df_inflation['year'] = df_inflation['observation_date'].str[:4].astype(int)

# 3. Fusionner les données sur l'année
df = pd.merge(df_chomage[['year', 'UNRTUKA_20171120']], 
              df_inflation[['year', 'CPIIUKA_20171120']], 
              on='year')

v_debut = 1948
v_fin = 1957

# 4. Filtrer pour la période d'étude de Phillips 
df_phillips = df[(df['year'] >= v_debut) & (df['year'] <= v_fin)].copy()
df_phillips = df_phillips.dropna()

# 5. Ajustement de la courbe par MCO (Polynomial de degré 2)
# La fonction np.polyfit calcule les coefficients du polynôme par MCO
x = df_phillips['UNRTUKA_20171120']
y = df_phillips['CPIIUKA_20171120']

# Calcul des coefficients (deg=2 pour une courbe)
coefficients = np.polyfit(x, y, 2)
polynomial = np.poly1d(coefficients)

# Génération des points prédits pour tracer la courbe lisse
x_range = np.linspace(x.min(), x.max(), 100)
y_pred = polynomial(x_range)

# 6. Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Données observées')
plt.plot(x_range, y_pred, color='red', label='Courbe de Phillips (MCO - Degré 2)')
plt.title(f"Courbe de Phillips au Royaume-Uni ({v_debut}-{v_fin}) - Ajustement MCO")
plt.xlabel("Taux de chômage (%)")
plt.ylabel("Taux d'inflation (%)")
plt.legend()
plt.grid(True)
plt.show()


y_hat = polynomial(x) 
y_mean = np.mean(y)
rss = np.sum((y - y_hat)**2)
tss = np.sum((y - y_mean)**2)
r2 = 1 - (rss / tss)

print(f"Le R² du modèle est : {r2:.4f}")