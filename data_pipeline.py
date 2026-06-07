"""
Amity University Online MCA Major Project
File: data_pipeline.py
Core Engine: Student Profiling Data Generation with Real Names, Roll Numbers, and IQR Handling
"""

import os
import pandas as pd
import numpy as np

def run_advanced_data_pipeline():
    print("=" * 80)
    print("[STEP 1] INITIALIZING ADVANCED STUDENT ARCHITECTURE PIPELINE...")
    print("=" * 80)
    
    # 1. Structural Concept Taxonomy
    concepts = ['Statistics', 'Linear_Algebra', 'SQL_Optimization', 'Predictive_Modeling', 'Model_Deployment']
    
    # Pool of Indian student names to replace generic User IDs
    student_names = [
        "Swapnita Paras", "Aarav Sharma", "Diya Iyer", "Kabir Malhotra", "Ananya Singh",
        "Rohan Verma", "Isha Choudhury", "Aditya Joshi", "Meera Nair", "Devanshu Gupta",
        "Kriti Saxena", "Arjun Kapoor", "Sneha Reddy", "Vivek Mishra", "Pooja Trivedi"
    ]
    
    # 2. Ingest Simulated Student Telemetry Logs
    np.random.seed(42)
    telemetry_space = []
    
    print("\n[STEP 2] GENERATING STUDENT PERFORMANCE AND LATENCY TELEMETRY MATRIX...")
    
    for idx, name in enumerate(student_names):
        roll_no = f"A99297240000{10 + idx:02d}"
        
        for concept in concepts:
            attempts = np.random.randint(1, 4)
            for attempt in range(attempts):
                base_correctness = 0.58 + (attempt * 0.08)
                correct = 1 if np.random.uniform(0, 1) < min(base_correctness, 0.95) else 0
                
                response_time = np.random.exponential(scale=18.0) if correct == 0 else np.random.normal(loc=14.0, scale=3.0)
                
                if np.random.uniform(0, 1) > 0.96:
                    response_time = np.random.uniform(180, 400)
                    
                telemetry_space.append({
                    'Student_Name': name,
                    'Roll_Number': roll_no,
                    'Concept_Tag': concept,
                    'Attempt_Sequence': attempt + 1,
                    'Item_Correctness': correct,
                    'Response_Latency': max(response_time, 1.0)
                })
                
    df_raw = pd.DataFrame(telemetry_space)
    print(f"✓ Raw Telemetry Logs Captured. Total Structural Rows: {df_raw.shape[0]}")
    
    # 3. Outlier Management Layer (IQR Method for Latency Cleaning)
    print("\n[STEP 3] APPLYING INTERQUARTILE RANGE (IQR) OUTLIER FILTERING...")
    q1 = df_raw['Response_Latency'].quantile(0.25)
    q3 = df_raw['Response_Latency'].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + (1.5 * iqr)
    
    df_raw['Cleaned_Latency'] = df_raw['Response_Latency'].clip(upper=upper_limit)
    print(f"✓ Outliers Treated. Maximum Latency capped at: {upper_limit:.2f} seconds")
    
    # 4. Feature Engineering & Aggregation
    print("\n[STEP 4] AGGREGATING BEHAVIORAL FEATURE VECTORS...")
    feature_aggregation = df_raw.groupby(['Student_Name', 'Roll_Number', 'Concept_Tag']).agg(
        Concept_Accuracy=('Item_Correctness', 'mean'),
        Mean_Latency=('Cleaned_Latency', 'mean'),
        Latency_Variance=('Cleaned_Latency', 'var'),
        Total_Attempt_Volume=('Attempt_Sequence', 'max')
    ).reset_index()
    
    feature_aggregation['Latency_Variance'] = feature_aggregation['Latency_Variance'].fillna(0.0)
    
    feature_aggregation['Path_Violations'] = np.where(
        (feature_aggregation['Concept_Tag'].isin(['Predictive_Modeling', 'Model_Deployment'])) & 
        (feature_aggregation['Concept_Accuracy'] > 0.75),
        np.random.poisson(lam=0.25, size=feature_aggregation.shape[0]),
        0
    )
    
    # 5. Export Cleaned Matrix for Streamlit
    print("\n[STEP 5] EXPORTING CLEANED MATRIX FOR MACHINE LEARNING PIPELINE...")
    os.makedirs('data', exist_ok=True)
    export_path = 'data/cleaned_movies_data.csv'
    feature_aggregation.to_csv(export_path, index=False)
    
    print("=" * 80)
    print(f"✅ PIPELINE SUCCESSFUL: Cleaned Matrix Saved to {export_path}")
    print("=" * 80)
    return feature_aggregation

if __name__ == "__main__":
    run_advanced_data_pipeline()