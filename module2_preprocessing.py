"""
==============================================================================
MODULE 2: Data Understanding & Preprocessing Pipeline
Data Mining Mini Project
Domain: EdTech / E-Learning Analytics
==============================================================================

WHY THIS MODULE MATTERS (Statistical Context):
    Raw educational data is messy and multi-formatted. We have numerical 
    metrics (watch time, scores) mixed with text-based transaction baskets 
    (course lists). Machine Learning models require strictly numerical, 
    scaled, and balanced inputs. 

    In this module we:
    1. Transform the course basket into One-Hot Encoded features.
    2. Normalize behavioral metrics using Z-score scaling so 'video_watch_mins' 
       (in thousands) doesn't overpower 'days_active' (in tens).
    3. Address class imbalance between dropouts and completions using SMOTE.

Dependencies: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn
Install: pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
==============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

# ── Output Folders ────────────────────────────────────────────────────────────
DATA_DIR = "outputs/data"
PLOTS_DIR = "outputs/plots/module2_preprocessing"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# ── Configure Plots ──────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("mako")

# ==============================================================================
# STEP 1: LOAD DATA & INITIAL INSPECTION
# ==============================================================================
print("=" * 70)
print("STEP 1: Loading Data & Initial Inspection")
print("=" * 70)

try:
    df = pd.read_csv(os.path.join(DATA_DIR, "student_learning_behavior.csv"))
    print(f"Dataset Loaded Successfully.")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
except FileNotFoundError:
    print("[ERROR] student_learning_behavior.csv not found. Please run the generator script.")
    exit()

# ==============================================================================
# STEP 2: MISSING VALUE ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Missing Value Analysis")
print("=" * 70)

missing = df.isnull().sum()
if missing.sum() == 0:
    print("  [OK] No missing values detected in any column.")
else:
    print(missing[missing > 0])
    df.fillna(df.median(numeric_only=True), inplace=True) # Basic imputation fallback

# ==============================================================================
# STEP 3: BASKET TRANSFORMATION & FEATURE ENGINEERING
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# The `enrolled_courses` column is a comma-separated string. Classifiers 
# (Decision Trees, k-NN) cannot read this. We must use `str.get_dummies()` 
# to create a multi-hot encoded matrix where each course becomes its own 
# binary column (1 if enrolled, 0 if not).
# 
# We also engineer a new feature: `total_courses_taken`, as course load 
# is highly correlated with dropout risk.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Basket Transformation & Feature Engineering")
print("=" * 70)

# 3a. Engineer course load feature
df['total_courses_taken'] = df['enrolled_courses'].apply(lambda x: len(x.split(',')))

# 3b. Multi-hot encode the course basket
print("  Transforming 'enrolled_courses' text into binary columns...")
courses_dummies = df['enrolled_courses'].str.get_dummies(sep=', ')

# Prefix the columns so we know they represent courses
courses_dummies = courses_dummies.add_prefix('course_')

# Merge back into the main dataframe
df = pd.concat([df, courses_dummies], axis=1)

# Drop the original text column and the ID column (not predictive)
cols_to_drop = ['student_id', 'enrolled_courses']
df.drop(columns=cols_to_drop, inplace=True)

print(f"  Dropped: {cols_to_drop}")
print(f"  New shape after engineering: {df.shape}")
print(f"  New feature columns generated: {list(courses_dummies.columns)}")

# ==============================================================================
# STEP 4: OUTLIER ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Outlier Analysis (Reporting Only)")
print("=" * 70)

numeric_cols = ['video_watch_mins', 'avg_quiz_score', 'days_active', 'total_courses_taken']
outlier_report = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = {'Outlier Count': n_outliers,
                           'Outlier %': round(n_outliers / len(df) * 100, 2)}

print(pd.DataFrame(outlier_report).T.to_string())
print("\n  [!] Decision: Outliers RETAINED. In education, extreme overachievers ")
print("      and extreme inactive students are exactly what we want to model.")

# ==============================================================================
# STEP 5: NORMALIZATION / SCALING
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# Distance-based algorithms (like k-NN for classification and k-Means for 
# clustering) are highly sensitive to magnitude. 
# Z-Score (StandardScaler): Maps data to Mean = 0, Std Dev = 1.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Normalization (Z-Score / StandardScaler)")
print("=" * 70)

scaler = StandardScaler()
# We only scale continuous metrics, NOT the binary course columns or the target
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(f"  Applied StandardScaler to: {numeric_cols}")

# ==============================================================================
# STEP 6: TRAIN/TEST SPLIT & SMOTE (Class Imbalance)
# ==============================================================================
# ─── Under the Hood ──────────────────────────────────────────────────────────
# We split the data BEFORE applying SMOTE. If we oversample the entire dataset 
# and then split it, synthetic data leaks into the test set, giving us falsely 
# high accuracy. 
# SMOTE interpolates between minority class points to create synthetic students, 
# ensuring our model learns to identify both dropouts and graduates equally.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Class Imbalance Analysis & SMOTE Resampling")
print("=" * 70)

# Check imbalance
class_counts = df['completion_status'].value_counts()
print("\n  Class Distribution (BEFORE resampling):")
print(f"    Dropouts (0):   {class_counts.get(0, 0):>5,d}")
print(f"    Completed (1):  {class_counts.get(1, 0):>5,d}")

# Separate features and target
X = df.drop(columns=['completion_status'])
y = df['completion_status']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

resampled_counts = pd.Series(y_train_resampled).value_counts()
print(f"\n  Class Distribution (AFTER SMOTE on training set):")
print(f"    Dropouts (0):   {resampled_counts[0]:>5,d}")
print(f"    Completed (1):  {resampled_counts[1]:>5,d}")
print("    Status: Balanced [OK]")

# ==============================================================================
# STEP 7: SAVE PREPROCESSED DATA & VISUALIZE
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Saving Data & Visualizations")
print("=" * 70)

# Reconstruct dataframes for saving
train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
train_resampled['completion_status'] = y_train_resampled.values

test_set = pd.DataFrame(X_test, columns=X_test.columns)
test_set['completion_status'] = y_test.values

# Save to CSV
train_resampled.to_csv(os.path.join(DATA_DIR, "preprocessed_train.csv"), index=False)
test_set.to_csv(os.path.join(DATA_DIR, "preprocessed_test.csv"), index=False)
df.to_csv(os.path.join(DATA_DIR, "preprocessed_full.csv"), index=False) # For clustering in Module 5

print(f"  Saved: {DATA_DIR}/preprocessed_train.csv  (SMOTE-balanced training set)")
print(f"  Saved: {DATA_DIR}/preprocessed_test.csv   (untouched test set for evaluation)")
print(f"  Saved: {DATA_DIR}/preprocessed_full.csv   (full dataset for clustering/pattern mining)")

# --- Generating Visualizations ---
# 1. Feature Correlation Heatmap (Behavioral Metrics + Target)
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = numeric_cols + ['completion_status']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Matrix: Student Behavior vs Completion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print(f"  Saved: {PLOTS_DIR}/correlation_heatmap.png")

print("\n" + "=" * 70)
print("MODULE 2 COMPLETE [OK]")
print("=" * 70)