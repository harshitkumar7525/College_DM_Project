"""
==============================================================================
MODULE 3: Pattern Mining (Apriori & FP-Growth)
Data Mining Mini Project
Domain: EdTech / E-Learning Analytics
==============================================================================

UNDER THE HOOD -- Why Pattern Mining on Mixed EdTech Data?
    In Phase 1, we applied Market Basket Analysis strictly to course names. 
    To match the advanced pipeline depth of our benchmark framework, we scale this 
    up by DISCRETIZING our numerical engagement metrics into categorical bins:
    
    Instead of: avg_quiz_score = 88.5
    We create:  avg_quiz_score = "quiz_HIGH"
    
    By transforming numbers into "items", we can discover powerful cross-domain 
    rules that combine behavior and curriculum, such as:
      {course_Machine Learning, watch_LOW, quiz_LOW} => {Status: Dropout}
      
    We also benchmark Apriori against FP-Growth to evaluate algorithmic 
    efficiency in terms of runtime execution and memory footprint.

Dependencies: pandas, numpy, mlxtend, time, tracemalloc
==============================================================================
"""

import os
import pandas as pd
import numpy as np
import time
import tracemalloc
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import warnings

# ── Output Folders ────────────────────────────────────────────────────────────
DATA_DIR = "outputs/data"
RESULTS_DIR = "outputs/results/module3_pattern_mining"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# ==============================================================================
# STEP 1: LOAD THE RAW DATA (PRE-SCALED)
# ==============================================================================
print("=" * 70)
print("STEP 1: Loading Raw Data for Discretization")
print("=" * 70)

try:
    df = pd.read_csv(os.path.join(DATA_DIR, "student_learning_behavior.csv"))
    print(f"  Loaded {len(df):,} student enrollment records.") # Matches Phase 1 total [cite: 16]
    print(f"  Target distribution (Completion Rate): {df['completion_status'].mean()*100:.2f}%")
except FileNotFoundError:
    print("[ERROR] student_learning_behavior.csv not found. Please run the data generator.")
    exit()

# ==============================================================================
# STEP 2: DISCRETIZE NUMERICAL FEATURES INTO CATEGORICAL BINS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Discretizing Behavioral Features into Domain Bins")
print("=" * 70)

df_disc = pd.DataFrame()

# --- video_watch_mins ---
df_disc['watch_time'] = pd.cut(df['video_watch_mins'],
    bins=[0, 300, 800, 1200, float('inf')],
    labels=['watch_LOW', 'watch_MED', 'watch_HIGH', 'watch_VHIGH'],
    include_lowest=True)

# --- avg_quiz_score ---
df_disc['quiz_score'] = pd.cut(df['avg_quiz_score'],
    bins=[0, 60, 75, 90, 100],
    labels=['quiz_FAIL', 'quiz_PASS', 'quiz_GOOD', 'quiz_EXCELLENT'],
    include_lowest=True)

# --- days_active ---
df_disc['platform_activity'] = pd.cut(df['days_active'],
    bins=[0, 7, 21, 45, float('inf')],
    labels=['act_LOW', 'act_MED', 'act_HIGH', 'act_VHIGH'],
    include_lowest=True)

# --- Target variable transformation into an Item ---
df_disc['status'] = df['completion_status'].map({0: 'status_DROPOUT', 1: 'status_COMPLETED'})

print("  Discretization complete. Sample binned behavioral items:")
print(df_disc.head(3))

# ==============================================================================
# STEP 3: ONE-HOT ENCODE INTO BOOLEAN BASKET FORMAT
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Creating One-Hot Boolean Basket Matrix")
print("=" * 70)

# 3a. Process the course basket strings (from Phase 1 approach) [cite: 22]
print("  One-hot encoding course items...")
course_basket = df['enrolled_courses'].str.get_dummies(sep=', ').astype(bool)
# Prefix course names for clarity
course_basket = course_basket.add_prefix('course_')

# 3b. One-hot encode our new discretized behavioral attributes
behavior_basket = pd.get_dummies(df_disc).astype(bool)

# 3c. Combine both baskets into a unified transactional matrix
basket = pd.concat([course_basket, behavior_basket], axis=1)

print(f"  Unified Basket Shape: {basket.shape[0]} rows, {basket.shape[1]} columns (items)")
print(f"  Total distinct item types available for mining: {basket.shape[1]}")

# ==============================================================================
# STEP 4: RUN APRIORI ALGORITHM
# ==============================================================================
# UNDER THE HOOD: Apriori generates candidates and filters iteratively level-by-level[cite: 26].
# We use a minimum support threshold of 0.02, mirroring Phase 2 expectations[cite: 25].
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Running Apriori Algorithm")
print("=" * 70)

tracemalloc.start()
t_start = time.time()

freq_apriori = apriori(basket, min_support=0.02, use_colnames=True, max_len=3)

t_apriori = time.time() - t_start
mem_apriori_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # Convert to MB
tracemalloc.stop()

print(f"  Apriori execution completed in {t_apriori:.4f} seconds")
print(f"  Peak Memory Consumed: {mem_apriori_peak:.4f} MB")
print(f"  Frequent itemsets discovered: {len(freq_apriori)}")

# ==============================================================================
# STEP 5: RUN FP-GROWTH ALGORITHM
# ==============================================================================
# UNDER THE HOOD: FP-Growth avoids heavy candidate generation steps by building 
# a highly compressed FP-Tree structure directly in memory using only 2 DB scans.
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Running FP-Growth Algorithm")
print("=" * 70)

tracemalloc.start()
t_start = time.time()

freq_fpgrowth = fpgrowth(basket, min_support=0.02, use_colnames=True, max_len=3)

t_fpgrowth = time.time() - t_start
mem_fpgrowth_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
tracemalloc.stop()

print(f"  FP-Growth execution completed in {t_fpgrowth:.4f} seconds")
print(f"  Peak Memory Consumed: {mem_fpgrowth_peak:.4f} MB")
print(f"  Frequent itemsets discovered: {len(freq_fpgrowth)}")

speedup = t_apriori / t_fpgrowth if t_fpgrowth > 0 else float('inf')
print(f"\n  --- Benchmark Analysis: FP-Growth is {speedup:.2f}x faster than Apriori ---")

# ==============================================================================
# STEP 6: GENERATE & FILTER ASSOCIATION RULES
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Generating Strategic Association Rules")
print("=" * 70)

# Generate rules using lift metric [cite: 7, 33]
rules = association_rules(freq_fpgrowth, metric="lift", min_threshold=1.2)
print(f"  Total baseline rules generated: {len(rules)}")

# 6a. FILTER TYPE A: Pure Course-to-Course Paths (Validating Phase 1 Results) [cite: 34]
course_rules = rules[
    rules['antecedents'].apply(lambda x: all(i.startswith('course_') for i in x)) &
    rules['consequents'].apply(lambda x: all(i.startswith('course_') for i in x))
]
course_rules = course_rules.sort_values('lift', ascending=False)

print("\n  TOP 8 CURRICULUM ROUTING RULES (Course -> Course):")
print("  " + "-" * 95)
print(f"  {'#':>3}  {'Antecedent (If Enrolled)}':<45} {'Consequent (Then Recomm)}':<30} {'Supp':>6} {'Conf':>6} {'Lift':>6}")
print("  " + "-" * 95)
for i, (_, row) in enumerate(course_rules.head(8).iterrows()):
    ant = ', '.join([i.replace('course_', '') for i in row['antecedents']])
    cons = ', '.join([i.replace('course_', '') for i in row['consequents']])
    print(f"  {i+1:>3}  {ant:<45} {cons:<30} {row['support']:>6.4f} {row['confidence']:>6.2%} {row['lift']:>6.2f}")

# 6b. FILTER TYPE B: Predictive Academic Risk Rules (Behavior -> Dropout Status)
dropout_rules = rules[
    rules['consequents'].apply(lambda x: 'status_status_DROPOUT' in x)
]
dropout_rules = dropout_rules.sort_values('lift', ascending=False)

print("\n  TOP 8 PREDICTIVE ACADEMIC RISK RULES (Behavior -> Dropout):")
print("  " + "-" * 95)
print(f"  {'#':>3}  {'Antecedent (Student Behavior Profile)}':<55} {'Consequent':<18} {'Supp':>6} {'Conf':>6} {'Lift':>6}")
print("  " + "-" * 95)
for i, (_, row) in enumerate(dropout_rules.head(8).iterrows()):
    ant = ', '.join([i.replace('status_', '').replace('course_', 'Enrolled: ') for i in row['antecedents']])
    cons = "Predict DROPOUT"
    print(f"  {i+1:>3}  {ant:<55} {cons:<18} {row['support']:>6.4f} {row['confidence']:>6.2%} {row['lift']:>6.2f}")

# ==============================================================================
# STEP 7: ALGORITHM COMPUTATIONAL COMPARISON TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Computational Complexity Analysis")
print("=" * 70)

comparison_metrics = {
    'Evaluation Dimension': ['Execution Time (sec)', 'Peak Memory Usage (MB)', 
                             'Frequent Itemsets Generated', 'Database Scan Cost', 
                             'Candidate Generation Phase?'],
    'Apriori Framework': [f'{t_apriori:.4f}', f'{mem_apriori_peak:.2f}', 
                          str(len(freq_apriori)), 'Iterative k-scans', 'Yes (Expensive)'],
    'FP-Growth Engine': [f'{t_fpgrowth:.4f}', f'{mem_fpgrowth_peak:.2f}', 
                        str(len(freq_fpgrowth)), '2 Scans (Constant)', 'No (Prefix-Tree)']
}
print(pd.DataFrame(comparison_metrics).to_string(index=False))

# ==============================================================================
# STEP 8: SAVE DATASETS TO CSV
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 8: Exporting Output Datasets")
print("=" * 70)

# Clean up set formatting strings for cleaner flat CSV file representation
rules_export = rules.copy()
rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))

rules_export.to_csv(os.path.join(RESULTS_DIR, "academic_association_rules.csv"), index=False)
print(f"  Successfully exported: {RESULTS_DIR}/academic_association_rules.csv")
print("\nMODULE 3 EXPORT PROCESSING COMPLETE [OK]")
print("=" * 70)