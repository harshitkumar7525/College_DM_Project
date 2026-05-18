"""
==============================================================================
MODULE 4: Classification Modeling & Evaluation
Data Mining Mini Project
Domain: EdTech / E-Learning Analytics
==============================================================================

WHY THIS MODULE MATTERS:
    We train supervised machine learning frameworks to predict whether a student
    will successfully complete their course track (completion_status = 1) or 
    drop out (completion_status = 0). 

    Because our baseline evaluation environment retains its original operational 
    imbalance, raw classification accuracy is heavily biased and highly misleading.
    Instead, we benchmark our algorithms using robust performance boundaries: 
    Precision, Recall, F1-Score, ROC-AUC, and Precision-Recall AUC curves.

CLASSIFIERS IMPLEMENTED:
1. Decision Tree Classifiers (Greedy Partitioning Framework)
2. Gaussian Naive Bayes (Probabilistic Feature Framework)
3. k-Nearest Neighbors (Instance-Based Distance Framework)

Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn
==============================================================================
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             auc, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# ── Configure Plots ──────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("plasma")

print("=" * 70)
print("MODULE 4: Classification Modeling Engine")
print("Algorithms: Decision Tree | Naive Bayes | k-NN")
print("=" * 70)

# ==============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ==============================================================================
print("\n[STEP 1] Loading Preprocessed Training and Test Partitions...")
try:
    df_train = pd.read_csv("preprocessed_train.csv")
    df_test = pd.read_csv("preprocessed_test.csv")
    print(f"  Training Vector Configuration (SMOTE Balanced) : {df_train.shape}")
    print(f"  Test Vector Configuration (Original Imbalance)  : {df_test.shape}")
except FileNotFoundError:
    print("[ERROR] Preprocessed data files missing. Please execute module2_preprocessing.py first.")
    exit()

X_train = df_train.drop(columns=['completion_status'])
y_train = df_train['completion_status']

X_test = df_test.drop(columns=['completion_status'])
y_test = df_test['completion_status']

print(f"  Total Predictive Operational Features: {X_train.shape[1]}")

# ==============================================================================
# STEP 2: MODEL INITIALIZATION WITH HYPERPARAMETER TUNING
# ==============================================================================
# UNDER THE HOOD - Hyperparameter Choices:
#   - Decision Tree (max_depth=6): Structural constraint added to prune deep, 
#     noisy nodes, mitigating variance spikes and overfitting.
#   - k-NN (n_neighbors=7): Chosen as an odd value to natively prevent voting ties
#     in binary classification setups.
# ==============================================================================
print("\n[STEP 2] Initializing Estimator Matrices with Justified Boundaries...")

models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=42
    ),
    "Naive Bayes (Gaussian)": GaussianNB(
        var_smoothing=1e-9
    ),
    "k-NN (k=7)": KNeighborsClassifier(
        n_neighbors=7,
        metric='euclidean',
        weights='distance',
        n_jobs=-1
    )
}

# ==============================================================================
# STEP 3: TRAINING AND PERFORMANCE EVALUATION
# ==============================================================================
print("\n[STEP 3] Running Pipeline Execution Blocks...")
print("-" * 70)

results = {}

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    print(f"\n  Execution Unit -> Processing: {name}")
    
    # Track Training Performance
    t_start = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t_start
    
    # Track Inference Performance
    t_pred_start = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    t_pred = time.time() - t_pred_start
    
    # Compute Exact Statistical Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    
    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 
        'F1-Score': f1, 'ROC-AUC': roc_auc, 'PR-AUC': pr_auc,
        'Train Time (s)': round(t_train, 4), 'Inference Time (s)': round(t_pred, 4)
    }
    
    print(f"     Metrics Status Summary: F1={f1:.4f} | Recall={rec:.4f} | Precision={prec:.4f}")
    
    # Plot Evaluation Trajectories
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
    ax_pr.plot(recall_vals, precision_vals, label=f"{name} (AUC = {pr_auc:.3f})", linewidth=2)
    
    # Build Confusion Matrices
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dropout', 'Completed'])
    disp.plot(ax=axes_cm[idx], cmap='Purples', values_format='d')
    axes_cm[idx].set_title(f"{name}\nF1-Score: {f1:.3f}", fontsize=11, fontweight='bold')
    axes_cm[idx].grid(False)

# ==============================================================================
# STEP 4: SAVE GRAPHICAL EVALUATIONS
# ==============================================================================
print("\n[STEP 4] Writing Evaluation Dashboards to Storage Layers...")

# Save ROC Trajectory
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guessing')
ax_roc.set_xlabel('False Positive Rate (FPR)', fontsize=11)
ax_roc.set_ylabel('True Positive Rate (TPR)', fontsize=11)
ax_roc.set_title('Receiver Operating Characteristic (ROC) Benchmark', fontsize=12, fontweight='bold')
ax_roc.legend(loc='lower right')
fig_roc.tight_layout()
fig_roc.savefig('classification_roc_curves.png', dpi=150)
plt.close(fig_roc)

# Save PR Trajectory
ax_pr.axhline(y=y_test.mean(), color='r', linestyle='--', alpha=0.5, label=f'Baseline ({y_test.mean():.3f})')
ax_pr.set_xlabel('Recall', fontsize=11)
ax_pr.set_ylabel('Precision', fontsize=11)
ax_pr.set_title('Precision-Recall Curve (PR-AUC) Space', fontsize=12, fontweight='bold')
ax_pr.legend(loc='lower left')
fig_pr.tight_layout()
fig_pr.savefig('classification_pr_curves.png', dpi=150)
plt.close(fig_pr)

# Save Matrices Group
fig_cm.suptitle('Confusion Matrix Evaluations (Imbalanced Test Set Target Boundary)', fontsize=13, fontweight='bold', y=1.02)
fig_cm.tight_layout()
fig_cm.savefig('classification_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close(fig_cm)

print("  Saved Visualizations: classification_roc_curves.png, classification_pr_curves.png, classification_confusion_matrices.png")

# ==============================================================================
# STEP 5: EXPLAINABLE AI (XAI) & STRUCTURAL INTERPRETABILITY
# ==============================================================================
print("\n[STEP 5] Extracting Structural Inductive Biases (Decision Tree Interpretability)...")

dt_estimator = models["Decision Tree"]

# Output Rules to Text Stream
print("\n  Top-Level Decision Logic Tree Split Structural Matrix:")
print("-" * 75)
text_rules = export_text(dt_estimator, feature_names=list(X_train.columns), max_depth=3)
print(text_rules)

# Save Feature Relative Importances
importances = dt_estimator.feature_importances_
feature_imp_df = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
sns.barplot(x=feature_imp_df.head(8).values, y=feature_imp_df.head(8).index, ax=ax_fi, palette="flare")
ax_fi.set_title('Top 8 Feature Importances (Gini Entropy Degradation Score)', fontsize=12, fontweight='bold')
ax_fi.set_xlabel('Gini Relative Importance Scale')
fig_fi.tight_layout()
fig_fi.savefig('classification_feature_importance.png', dpi=150)
plt.close(fig_fi)
print("  Saved Feature Importance Chart: classification_feature_importance.png")

# ==============================================================================
# STEP 6: SENSITIVITY HIGHER-ORDER ANALYSIS (k Optimization)
# ==============================================================================
print("\n[STEP 6] Executing Neighbor Sensitivity Probes (Varying Hyperparameter k)...")

k_search_space = [3, 5, 7, 9, 11]
sensitivity_metrics = []

for k in k_search_space:
    knn_probe = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance', n_jobs=-1)
    knn_probe.fit(X_train, y_train)
    probe_preds = knn_probe.predict(X_test)
    
    sensitivity_metrics.append({
        'k_value': k,
        'Precision': precision_score(y_test, probe_preds),
        'Recall': recall_score(y_test, probe_preds),
        'F1-Score': f1_score(y_test, probe_preds)
    })

sensitivity_df = pd.DataFrame(sensitivity_metrics).set_index('k_value')
print(sensitivity_df.round(4))

# Save Sensitivity Analysis Graph
fig_k, ax_k = plt.subplots(figsize=(8, 4.5))
sensitivity_df.plot(ax=ax_k, marker='o', linewidth=2)
ax_k.set_xlabel('Hyperparameter k Value (Number of Overlap Boundaries)')
ax_k.set_ylabel('Operational Score Metrics')
ax_k.set_title('k-NN Parametric Sensitivity Profile Matrix', fontsize=12, fontweight='bold')
ax_k.set_xticks(k_search_space)
fig_k.tight_layout()
fig_k.savefig('classification_knn_sensitivity.png', dpi=150)
plt.close(fig_k)
print("  Saved Hyperparameter Sweep Curve: classification_knn_sensitivity.png")

# ==============================================================================
# STEP 7: MASTER COMPARATIVE PERFORMANCE MATRIX
# ==============================================================================
print("\n" + "=" * 70)
print("MODULE 4 COMPLETE: FINAL MODEL COMPARISON MATRIX")
print("=" * 70)

final_report_df = pd.DataFrame(results).T
print(final_report_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']].round(4).to_string())

print("\nComputational Cost Analysis:")
print("-" * 50)
print(final_report_df[['Train Time (s)', 'Inference Time (s)']].to_string())

# Save metrics record to disk storage
final_report_df.to_csv("classification_metrics_summary.csv")
print("\n  CSV Metric Logs Saved Successfully: classification_metrics_summary.csv\n")