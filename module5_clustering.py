"""
==============================================================================
MODULE 5: Clustering & Anomaly Detection
Data Mining Mini Project
Domain: EdTech / E-Learning Analytics
==============================================================================

WHY THIS MODULE MATTERS:
    Unlike supervised classification, clustering operates without any target 
    labels (unsupervised). The engine's objective is to scan student features 
    and identify latent structural groupings or archetypes naturally present 
    within the platform.

    By cross-tabulating these discovered clusters against the hidden ground-truth 
    completion rates, platform administrators can uncover behavioral cohorts 
    (e.g., "High-Yield Achievers" vs "At-Risk Crammers") to drive automated, 
    personalized student support systems.

ALGORITHMS IMPLEMENTED:
1. k-Means Partitioning (Centroid Minimization)
2. Hierarchical Agglomerative Clustering (Ward Linkage Variance Minimization)
3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DIMENSIONALITY REDUCTION:
    - PCA (Principal Component Analysis): Linear variance maximization.
    - t-SNE (t-Distributed Stochastic Neighbor Embedding): Non-linear neighborhood mapping.

Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy
==============================================================================
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# ── Configure Plots ──────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

print("=" * 70)
print("MODULE 5: Unsupervised Clustering & Anomaly Engine")
print("Algorithms: k-Means | Hierarchical Agglomerative | DBSCAN")
print("=" * 70)

# ==============================================================================
# STEP 1: LOAD UNBALANCED FULL PREPROCESSED DATA
# ==============================================================================
print("\n[STEP 1] Loading Preprocessed Operational Space...")
try:
    df = pd.read_csv("preprocessed_full.csv")
    print(f"  Full Matrix Workspace Dimension: {df.shape[0]} rows, {df.shape[1]} features")
except FileNotFoundError:
    print("[ERROR] preprocessed_full.csv missing. Please run module2_preprocessing.py first.")
    exit()

# Extract hidden ground truth labels for analytical comparison later
y_true = df['completion_status'].values
X = df.drop(columns=['completion_status'])

# ==============================================================================
# STEP 2: SUBSAMPLING FOR COMPUTATIONAL EFFICIENCY
# ==============================================================================
# UNDER THE HOOD: Hierarchical clustering requires computing an O(N^2) memory 
# linkage matrix, and t-SNE scales quadratically. We draw a representative 
# stratified sample of 3,000 students to keep visualization rendering times snappy.
# ==============================================================================
print("\n[STEP 2] Extracting Downstream Analytical Slice (Sampling)...")
SAMPLE_SIZE = 3000

np.random.seed(42)
sample_indices = np.random.choice(len(X), SAMPLE_SIZE, replace=False)

X_sample = X.iloc[sample_indices].reset_index(drop=True)
y_sample = y_true[sample_indices]

print(f"  Sampled Space Sliced: {X_sample.shape[0]} Rows")

# ==============================================================================
# STEP 3: DIMENSIONALITY REDUCTION (PCA vs t-SNE)
# ==============================================================================
print("\n[STEP 3] Computing Low-Dimensional Projections (PCA & t-SNE)...")

# 3a. Principal Component Analysis (PCA)
pca_engine = PCA(n_components=2, random_state=42)
X_pca = pca_engine.fit_transform(X_sample)
var_ratio = pca_engine.explained_variance_ratio_
print(f"  PCA Mapping Complete: Total Explained Variance = {var_ratio.sum()*100:.2f}% (PC1={var_ratio[0]*100:.1f}%)")

# 3b. t-Distributed Stochastic Neighbor Embedding (t-SNE)
print("  Fitting Non-Linear Local Neighborhood Embeddings (t-SNE)...")
t_start = time.time()
# Standardized parameter 'max_iter' implemented here for newer scikit-learn versions
tsne_engine = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=800)
X_tsne = tsne_engine.fit_transform(X_sample)
print(f"  t-SNE Embedding Extracted in {time.time() - t_start:.2f} seconds.")

# ==============================================================================
# STEP 4: K-MEANS PARTITIONING ANALYSIS
# ==============================================================================
print("\n[STEP 4] Executing Centroid Convergence (k-Means)...")

k_space = range(2, 8)
inertias = []
silhouettes = []

for k in k_space:
    km_test = KMeans(n_clusters=k, n_init=10, random_state=42)
    test_labels = km_test.fit_predict(X_sample)
    inertias.append(km_test.inertia_)
    silhouettes.append(silhouette_score(X_sample, test_labels, sample_size=1500))

# Save Elbow/Silhouette Curves
fig_eval, axes_eval = plt.subplots(1, 2, figsize=(12, 4))
axes_eval[0].plot(k_space, inertias, 'bo-', linewidth=2)
axes_eval[0].set_title('Elbow Optimization Interface (Inertia)', fontweight='bold')
axes_eval[0].set_xlabel('Centroid Count (k)')
axes_eval[1].plot(k_space, silhouettes, 'ro-', linewidth=2)
axes_eval[1].set_title('Silhouette Cohort Separability Profile', fontweight='bold')
axes_eval[1].set_xlabel('Centroid Count (k)')
fig_eval.tight_layout()
fig_eval.savefig('clustering_optimization_curves.png', dpi=150)
plt.close()

# Run optimal deployment at k=3 based on behavior structures
OPTIMAL_K = 3
km_final = KMeans(n_clusters=OPTIMAL_K, n_init=10, random_state=42)
km_labels = km_final.fit_predict(X_sample)
km_sil = silhouette_score(X_sample, km_labels)

print(f"  k-Means Configuration Complete (k={OPTIMAL_K}) -> Silhouette Coefficient: {km_sil:.4f}")

# ==============================================================================
# STEP 5: AGGLOMERATIVE HIERARCHICAL CLUSTERING
# ==============================================================================
print("\n[STEP 5] Building Agglomerative Structural Dendrogram Traces...")

# Compute Ward Linkage on a clean micro-slice for visual clear formatting
Z_matrix = linkage(X_sample.head(300), method='ward')

fig_dendro, ax_dendro = plt.subplots(figsize=(12, 5))
dendrogram(Z_matrix, truncate_mode='lastp', p=20, ax=ax_dendro, leaf_rotation=90)
ax_dendro.set_title('Agglomerative Hierarchical Topology (Ward Variance Linkage)', fontsize=12, fontweight='bold')
fig_dendro.tight_layout()
fig_dendro.savefig('clustering_dendrogram_tree.png', dpi=150)
plt.close()

# Fit target structure
agg_engine = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward')
agg_labels = agg_engine.fit_predict(X_sample)
agg_sil = silhouette_score(X_sample, agg_labels)
print(f"  Hierarchical Slices Locked (Clusters={OPTIMAL_K}) -> Silhouette Coefficient: {agg_sil:.4f}")

# ==============================================================================
# STEP 6: DENSITY-BASED SPATIAL ANOMALY DETECTION (DBSCAN)
# ==============================================================================
# UNDER THE HOOD: DBSCAN handles tracking arbitrary shapes and isolating outliers.
# Points failing to meet density connectivity indices fall into Noise Cluster (-1).
# ==============================================================================
print("\n[STEP 6] Deploying Density-Based Anomaly Isolation (DBSCAN)...")

# Hyperparameter parameters chosen via distance knee evaluation metrics 
dbscan_engine = DBSCAN(eps=2.2, min_samples=8, n_jobs=-1)
db_labels = dbscan_engine.fit_predict(X_sample)

discovered_cores = len(set(db_labels)) - (1 if -1 in db_labels else 0)
noise_count = (db_labels == -1).sum()

print(f"  DBSCAN Scan Processing Complete: Found {discovered_cores} High-Density Cores.")
print(f"  Isolated Anomalous Outliers (Labeled Noise: -1): {noise_count} points ({noise_count/len(X_sample)*100:.1f}%)")

# ==============================================================================
# STEP 7: INTERPRETIVE VALIDATION CROSS-TABULATIONS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Latent Behavior-to-Success Alignment (Validation Tables)")
print("=" * 70)

print("\n  [Table A] k-Means Structural Clusters vs Student Completion Performance:")
ct_km = pd.crosstab(km_labels, y_sample, rownames=['Cluster ID'], colnames=['Completion Status'])
ct_km.columns = ['Dropped Out', 'Completed Track']
ct_km['Observed_Success_Rate_%'] = (ct_km['Completed Track'] / (ct_km['Completed Track'] + ct_km['Dropped Out']) * 100).round(2)
print(ct_km.to_string())

print("\n  [Table B] DBSCAN Anomaly Core Clusters vs Student Completion Performance:")
ct_db = pd.crosstab(db_labels, y_sample, rownames=['DBSCAN Core ID'], colnames=['Completion Status'])
ct_db.columns = ['Dropped Out', 'Completed Track']
if -1 in ct_db.index:
    ct_db.rename(index={-1: 'Anomalies / Noise (-1)'}, inplace=True)
ct_db['Observed_Success_Rate_%'] = (ct_db['Completed Track'] / (ct_db['Completed Track'] + ct_db['Dropped Out']) * 100).round(2)
print(ct_db.to_string())

# ==============================================================================
# STEP 8: MASTER COHORT VISUALIZATION COMPONENT
# ==============================================================================
print("\n[STEP 8] Constructing Twin Projection Spatial Cluster Maps...")

fig_maps, axes_maps = plt.subplots(2, 3, figsize=(18, 10))
labels_group = [km_labels, agg_labels, db_labels]
engine_names = ['k-Means Space', 'Hierarchical Ward Space', 'DBSCAN Density Space']

for col_idx, (lbls, title) in enumerate(zip(labels_group, engine_names)):
    # Row 0: PCA Visualizations
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=lbls, palette='turbo', alpha=0.6, s=15, ax=axes_maps[0, col_idx], legend='brief')
    axes_maps[0, col_idx].set_title(f"{title} (PCA Axis)", fontweight='bold')
    axes_maps[0, col_idx].set_xlabel('Principal Component 1')
    axes_maps[0, col_idx].set_ylabel('Principal Component 2')
    
    # Row 1: t-SNE Visualizations
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=lbls, palette='turbo', alpha=0.6, s=15, ax=axes_maps[1, col_idx], legend='brief')
    axes_maps[1, col_idx].set_title(f"{title} (t-SNE Coordinate)", fontweight='bold')
    axes_maps[1, col_idx].set_xlabel('Embedding Axis 1')
    axes_maps[1, col_idx].set_ylabel('Embedding Axis 2')

fig_maps.suptitle('Operational Cluster Matrix Topographies (PCA vs t-SNE Projections)', fontsize=15, fontweight='bold', y=0.98)
fig_maps.tight_layout()
fig_maps.savefig('clustering_master_topographies.png', dpi=150)
plt.close()

print("  Saved Master Structural Maps: clustering_master_topographies.png")

# ==============================================================================
# STEP 9: WRITE EVALUATION RESULTS TO FILES
# ==============================================================================
summary_metrics = {
    'Unsupervised Model Asset': ['k-Means Partitioning', 'Agglomerative Ward Tree', 'DBSCAN Core Engine'],
    'Discovered Clusters': [OPTIMAL_K, OPTIMAL_K, discovered_cores],
    'Isolations / Noise Matches': [0, 0, noise_count],
    'Silhouette Separation Score': [round(km_sil, 4), round(agg_sil, 4), 
                                     round(silhouette_score(X_sample[db_labels!=-1], db_labels[db_labels!=-1]), 4) if discovered_cores > 1 else 0.0],
    'Adjusted Rand Score (ARI)': [round(adjusted_rand_score(y_sample, km_labels), 4), 
                                  round(adjusted_rand_score(y_sample, agg_labels), 4), 
                                  round(adjusted_rand_score(y_sample, db_labels), 4)]
}

summary_df = pd.DataFrame(summary_metrics)
summary_df.to_csv("clustering_metrics_summary.csv", index=False)
print("\n[MODULE 5 STATUS] Clustering Engine Execution Pipeline Successfully Concluded [OK]\n")