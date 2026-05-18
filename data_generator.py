import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# 1. Define the 10 courses from Team 11 - Phase 1 Report
COURSES = [
    "Python Programming", "Machine Learning", "Data Science", 
    "Artificial Intelligence", "Cloud Computing", "Deep Learning", 
    "Data Visualization", "Cyber Security", "Blockchain", "Web Development"
]

NUM_STUDENTS = 6200 # Matching the report's statistics

data = []

for student_id in range(1, NUM_STUDENTS + 1):
    # --- Simulate Course Enrollments (Basket Data) ---
    # Average courses per student is roughly 2.8 according to the report
    num_enrolled = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.30, 0.15, 0.05])
    enrolled_courses = random.sample(COURSES, num_enrolled)
    
    # Inject Phase 1 Rules (Force correlation between ML and Deep Learning)
    if "Deep Learning" in enrolled_courses and "Machine Learning" not in enrolled_courses:
        if random.random() < 0.60: # 60% chance to add ML if they have DL
            enrolled_courses.append("Machine Learning")
    
    # --- Simulate Numerical / Behavioral Features (For Clustering/Classification) ---
    # Create two 'types' of students to make clustering interesting
    is_dedicated = random.random() > 0.4 
    
    if is_dedicated:
        video_watch_mins = int(np.random.normal(1200, 300))
        avg_quiz_score = np.random.normal(85, 10)
        days_active = int(np.random.normal(45, 15))
        completion_status = 1 if random.random() > 0.1 else 0 # 90% chance to complete
    else:
        video_watch_mins = int(np.random.normal(300, 150))
        avg_quiz_score = np.random.normal(55, 15)
        days_active = int(np.random.normal(10, 5))
        completion_status = 1 if random.random() > 0.85 else 0 # 15% chance to complete

    # Clip values to realistic ranges
    video_watch_mins = max(10, video_watch_mins)
    avg_quiz_score = min(100, max(0, avg_quiz_score))
    days_active = max(1, days_active)
    
    # Store record
    data.append({
        "student_id": f"STU_{student_id:05d}",
        "enrolled_courses": ", ".join(enrolled_courses),
        "video_watch_mins": video_watch_mins,
        "avg_quiz_score": round(avg_quiz_score, 1),
        "days_active": days_active,
        "completion_status": completion_status
    })

# Create DataFrame and save
df = pd.DataFrame(data)

# ── Output Folder ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_csv(os.path.join(OUTPUT_DIR, "student_learning_behavior.csv"), index=False)

print(f"Dataset generated successfully with {len(df)} records!")
print(f"Saved to: {OUTPUT_DIR}/student_learning_behavior.csv")
print(df.head())