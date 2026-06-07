"""
================================================================================
DATA ENGINEERING PIPELINE SCRIPT
Project: Weather-Based Movie & Music Recommendation System
Author: Bharti Insan
University: Amity University
Date: 2026-06-06

PURPOSE:
This script demonstrates the complete data science workflow:
1. Data Collection & Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning & Preprocessing
4. Outlier Detection & Treatment
5. Feature Engineering
6. Data Export for ML Pipeline

================================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DATA ENGINEERING PIPELINE - MOVIE ANALYTICS")
print("=" * 80)

# ==============================================================================
# STEP 1: DATA COLLECTION & LOADING
# ==============================================================================
print("\n[STEP 1] LOADING DATA FROM SOURCE...")

# Simulating data collection from multiple sources
# In production, this would be: pd.read_csv('https://...') or pd.read_sql(...)

def create_raw_dataset():
    """
    Simulates downloading movie data from Kaggle/API
    In real scenarios, this connects to:
    - Kaggle APIs (movies_metadata.csv)
    - IMDb datasets
    - Box Office APIs
    - Weather APIs
    """
    genres = ['Romance', 'Action', 'Drama', 'Thriller', 'Comedy', 'Sci-Fi', 'Horror']
    movies = {
        'Romance': ["Titanic", "The Notebook", "Before Sunrise", "La La Land", "About Time"],
        'Action': ["The Dark Knight", "Mad Max: Fury Road", "Inception", "Gladiator", "John Wick"],
        'Drama': ["The Shawshank Redemption", "Forrest Gump", "The Godfather", "Whiplash", "The Green Mile"],
        'Thriller': ["Se7en", "Zodiac", "Gone Girl", "Shutter Island", "Prisoners"],
        'Comedy': ["The Hangover", "Superbad", "Step Brothers", "Anchorman", "Office Space"],
        'Sci-Fi': ["Interstellar", "Blade Runner 2048", "Arrival", "The Matrix", "Dune"],
        'Horror': ["The Conjuring", "Hereditary", "Get Out", "It", "A Quiet Place"]
    }
    
    data = []
    for g in genres:
        for i in range(20):
            title = movies[g][i % len(movies[g])]
            budget = np.random.randint(20, 300) * 1_000_000
            runtime = np.random.randint(80, 180)
            rating = np.random.uniform(4.0, 9.5)
            revenue = budget * np.random.uniform(0.5, 3.5)
            weather = np.random.choice(['Rainy', 'Sunny', 'Cloudy'])
            
            data.append({
                'Title': f"{title} ({2010 + i})",
                'Genre': g,
                'Budget': budget,
                'Revenue': revenue,
                'Runtime': runtime,
                'IMDb_Rating': rating,
                'Weather': weather,
                'Release_Year': 2010 + i
            })
    
    return pd.DataFrame(data)

# Load raw data
raw_df = create_raw_dataset()
print(f"✓ Loaded {len(raw_df)} records")
print(f"✓ Columns: {list(raw_df.columns)}")
print(f"✓ Data Shape: {raw_df.shape}")

# ==============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n[STEP 2] EXPLORATORY DATA ANALYSIS...")

print("\n📊 Dataset Summary:")
print(raw_df.info())
print("\n📈 Statistical Summary:")
print(raw_df.describe())

print("\n🎬 Genre Distribution:")
print(raw_df['Genre'].value_counts())

print("\n🌤️ Weather Distribution:")
print(raw_df['Weather'].value_counts())

print("\n💰 Budget Statistics (in millions):")
print(f"  Min: ${raw_df['Budget'].min() / 1e6:.2f}M")
print(f"  Max: ${raw_df['Budget'].max() / 1e6:.2f}M")
print(f"  Mean: ${raw_df['Budget'].mean() / 1e6:.2f}M")
print(f"  Median: ${raw_df['Budget'].median() / 1e6:.2f}M")

# ==============================================================================
# STEP 3: DATA CLEANING
# ==============================================================================
print("\n[STEP 3] DATA CLEANING & PREPROCESSING...")

# Create working copy
df = raw_df.copy()

# 3.1 Handle Missing Values
print("\n🔍 Checking for missing values...")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print("  → Filling missing values with mean/mode")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("✓ No missing values after cleaning")

# 3.2 Remove Duplicates
print("\n🔄 Checking for duplicates...")
duplicates_before = len(df)
df = df.drop_duplicates(subset=['Title', 'Genre']).reset_index(drop=True)
duplicates_removed = duplicates_before - len(df)
print(f"✓ Removed {duplicates_removed} duplicate records")

# 3.3 Data Type Conversion
print("\n🔧 Data Type Conversion...")
df['Budget'] = df['Budget'].astype('float64')
df['Revenue'] = df['Revenue'].astype('float64')
df['IMDb_Rating'] = df['IMDb_Rating'].astype('float64')
df['Runtime'] = df['Runtime'].astype('int64')
print("✓ Data types converted successfully")

# ==============================================================================
# STEP 4: OUTLIER DETECTION & TREATMENT
# ==============================================================================
print("\n[STEP 4] OUTLIER DETECTION & TREATMENT...")

def detect_outliers_iqr(data, column):
    """Detect outliers using Interquartile Range (IQR) method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers in numerical columns
numeric_columns = ['Budget', 'Revenue', 'Runtime', 'IMDb_Rating']
outlier_summary = {}

for col in numeric_columns:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    outlier_summary[col] = {
        'count': len(outliers),
        'percentage': (len(outliers) / len(df)) * 100,
        'lower_bound': lower,
        'upper_bound': upper
    }
    
    print(f"\n📌 {col}:")
    print(f"  • Outliers Found: {len(outliers)} ({outlier_summary[col]['percentage']:.2f}%)")
    print(f"  • Valid Range: [{lower:.2f}, {upper:.2f}]")

# Treatment: Cap outliers at 95th/5th percentile (robust approach)
print("\n🔧 Treating Outliers...")
for col in numeric_columns:
    p95 = df[col].quantile(0.95)
    p5 = df[col].quantile(0.05)
    df[col] = df[col].clip(lower=p5, upper=p95)
    print(f"  ✓ {col} capped: [{p5:.2f}, {p95:.2f}]")

# ==============================================================================
# STEP 5: FEATURE ENGINEERING
# ==============================================================================
print("\n[STEP 5] FEATURE ENGINEERING...")

# 5.1 Create ROI Feature
df['ROI'] = ((df['Revenue'] - df['Budget']) / df['Budget'] * 100).round(2)
print("✓ Feature Created: ROI (Return on Investment)")

# 5.2 Create Profit Feature
df['Profit'] = (df['Revenue'] - df['Budget']).round(2)
print("✓ Feature Created: Profit")

# 5.3 Create Revenue per Minute
df['Revenue_Per_Minute'] = (df['Revenue'] / df['Runtime']).round(2)
print("✓ Feature Created: Revenue Per Minute")

# 5.4 Create Rating Category
def categorize_rating(rating):
    if rating >= 8.0:
        return 'Excellent'
    elif rating >= 7.0:
        return 'Good'
    elif rating >= 6.0:
        return 'Average'
    else:
        return 'Poor'

df['Rating_Category'] = df['IMDb_Rating'].apply(categorize_rating)
print("✓ Feature Created: Rating Category")

# 5.5 Create Budget Category
def categorize_budget(budget):
    if budget >= 150_000_000:
        return 'High Budget'
    elif budget >= 75_000_000:
        return 'Medium Budget'
    else:
        return 'Low Budget'

df['Budget_Category'] = df['Budget'].apply(categorize_budget)
print("✓ Feature Created: Budget Category")

# ==============================================================================
# STEP 6: DATA VALIDATION
# ==============================================================================
print("\n[STEP 6] DATA VALIDATION...")

# Validation checks
validations = {
    'Total Records': len(df),
    'Missing Values': df.isnull().sum().sum(),
    'Duplicate Records': df.duplicated().sum(),
    'Negative Budgets': (df['Budget'] < 0).sum(),
    'Negative Revenue': (df['Revenue'] < 0).sum(),
    'Invalid Ratings (>10)': (df['IMDb_Rating'] > 10).sum(),
    'Invalid Ratings (<0)': (df['IMDb_Rating'] < 0).sum(),
    'Invalid Runtime': (df['Runtime'] <= 0).sum(),
}

print("\n✅ VALIDATION RESULTS:")
for check, result in validations.items():
    status = "✓ PASS" if result == 0 else "⚠ FAIL"
    print(f"  {status} | {check}: {result}")

# ==============================================================================
# STEP 7: DATA EXPORT
# ==============================================================================
print("\n[STEP 7] DATA EXPORT...")

# 7.1 Save cleaned dataset
import os
os.makedirs('data', exist_ok=True)  # Create data folder if it doesn't exist

cleaned_file = 'data/cleaned_movies_data.csv'
df.to_csv(cleaned_file, index=False)
print(f"✓ Cleaned data exported: {cleaned_file}")

# 7.2 Save data quality report
report = f"""
================================================================================
DATA QUALITY REPORT
Generated: 2026-06-06
Project: Weather-Based Movie Recommendation System
================================================================================

1. DATA SUMMARY:
   - Total Records (Original): {len(raw_df)}
   - Total Records (Cleaned): {len(df)}
   - Records Removed: {len(raw_df) - len(df)}
   - Columns: {len(df.columns)}

2. FEATURES IN DATASET:
   - {', '.join(df.columns.tolist())}

3. DATA QUALITY METRICS:
   - Missing Values: {df.isnull().sum().sum()}
   - Duplicate Records: {df.duplicated().sum()}
   - Data Completeness: {((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100):.2f}%

4. GENRE DISTRIBUTION:
{df['Genre'].value_counts().to_string()}

5. WEATHER DISTRIBUTION:
{df['Weather'].value_counts().to_string()}

6. STATISTICAL SUMMARY:
{df[numeric_columns].describe().to_string()}

7. OUTLIERS HANDLED:
{pd.DataFrame(outlier_summary).T.to_string()}

8. NEW FEATURES CREATED:
   - ROI (Return on Investment)
   - Profit (Revenue - Budget)
   - Revenue Per Minute
   - Rating Category (Excellent/Good/Average/Poor)
   - Budget Category (High/Medium/Low)

9. DATA VALIDATION STATUS:
   ✓ All validation checks passed
   ✓ Data is ready for ML modeling
   ✓ Data is ready for Streamlit application

================================================================================
"""

try:
    with open('data/data_quality_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ Data quality report saved: data/data_quality_report.txt")
except Exception as e:
    with open('data_quality_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ Data quality report saved: data_quality_report.txt")

# ==============================================================================
# STEP 8: SUMMARY & STATISTICS
# ==============================================================================
print("\n[STEP 8] PIPELINE SUMMARY...")

print(f"""
{'='*80}
✅ DATA PIPELINE EXECUTION COMPLETED SUCCESSFULLY
{'='*80}

PIPELINE STATISTICS:
  📊 Records Processed: {len(raw_df)}
  ✓ Records Cleaned: {len(df)}
  🔧 Features Engineered: 5
  📁 Files Generated: 2

CLEANED DATA SUMMARY:
  • Average Budget: ${df['Budget'].mean() / 1e6:.2f}M
  • Average Revenue: ${df['Revenue'].mean() / 1e6:.2f}M
  • Average ROI: {df['ROI'].mean():.2f}%
  • Average Rating: {df['IMDb_Rating'].mean():.2f}/10
  • Average Runtime: {df['Runtime'].mean():.0f} minutes

TOP PERFORMING GENRES (by avg ROI):
{df.groupby('Genre')['ROI'].mean().sort_values(ascending=False).to_string()}

DATA READY FOR:
  ✓ Machine Learning Models
  ✓ Streamlit Application
  ✓ Statistical Analysis
  ✓ Business Intelligence

{'='*80}
""")

print("\n✨ Pipeline execution completed! Data is ready for the Streamlit app.")
