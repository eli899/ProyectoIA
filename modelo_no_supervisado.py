import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. CARGAR DATASET 
df = pd.read_csv('train.csv')
print(f"Dataset cargado: {df.shape}")
print(df.head())

# ── 2. PREPROCESAMIENTO 

df.dropna(inplace=True)

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# ── 3. ESTANDARIZACIÓN 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. MÉTODO DEL CODO 
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(list(K_range), inertias, 'o-', color='#185FA5',
         linewidth=2, markersize=7)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo — K-Means', fontsize=13)
plt.xticks(list(K_range))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('grafica_codo.png', dpi=200)
plt.show()
print("✓ Guardada: grafica_codo.png")

# ── 5. ENTRENAR CON K ÓPTIMO 
k_optimo = 5
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df['Cluster'])
print(f"\n{'='*35}")
print(f"  Silhouette Score (k={k_optimo}): {sil:.4f}")
print(f"  Inercia final: {kmeans.inertia_:.2f}")
print(f"{'='*35}")

# ── 6. RESUMEN POR CLUSTER 
resumen = df.groupby('Cluster')[
    ['Age','Annual Income (k$)','Spending Score (1-100)']
].mean().round(1)
print("\nResumen por cluster:")
print(resumen)

# ── 7. GRÁFICA: Ingreso vs Gasto por cluster 
colores = ['#378ADD','#1D9E75','#D85A30','#9b59b6','#BA7517']
plt.figure(figsize=(8, 6))
for i in range(k_optimo):
    mask = df['Cluster'] == i
    plt.scatter(df.loc[mask,'Annual Income (k$)'],
                df.loc[mask,'Spending Score (1-100)'],
                c=colores[i], label=f'Cluster {i+1}',
                s=80, alpha=0.85, edgecolors='white', linewidths=0.5)
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids_orig[:,1], centroids_orig[:,2],
            c='black', marker='X', s=200, zorder=5, label='Centroides')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntuación de Gasto (1-100)')
plt.title('Segmentación de Clientes — K-Means (k=5)', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_clusters.png', dpi=200)
plt.show()
print(" Guardada: grafica_clusters.png")

# ── 8. GRÁFICA PCA 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_
print(f"\nVarianza explicada PCA: CP1={var[0]*100:.1f}% | CP2={var[1]*100:.1f}%")

plt.figure(figsize=(8, 6))
for i in range(k_optimo):
    mask = df['Cluster'].values == i
    plt.scatter(X_pca[mask,0], X_pca[mask,1],
                c=colores[i], label=f'Cluster {i+1}',
                s=80, alpha=0.85, edgecolors='white', linewidths=0.5)
plt.xlabel(f'CP1 ({var[0]*100:.1f}% varianza)')
plt.ylabel(f'CP2 ({var[1]*100:.1f}% varianza)')
plt.title('Clusters visualizados con PCA', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('grafica_pca.png', dpi=200)
plt.show()
print(" Guardada: grafica_pca.png")