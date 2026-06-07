"""
Amity University Online Major Project Portfolio
File: app.py
Platform: Streamlit Dashboard with Random Forest Classification Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestClassifier

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(
    page_title="AI Capability Analytics Engine",
    page_icon="🧠",
    layout="wide"
)

# Initialize Session States for Navigation Buttons
for state_key in ['render_recommendations', 'render_cohort_trends']:
    if state_key not in st.session_state:
        st.session_state[state_key] = False

# --- AUTOMATED PIPELINE SYNC & INGESTION ---
@st.cache_data
def load_educational_matrix():
    data_path = 'data/cleaned_movies_data.csv'
    if not os.path.exists(data_path):
        from data_pipeline import run_advanced_data_pipeline
        return run_advanced_data_pipeline()
    return pd.read_csv(data_path)

df_analytics = load_educational_matrix()

# --- MACHINE LEARNING MODEL COMPILE (Random Forest Engine) ---
@st.cache_resource
def train_rf_classifier(data):
    X = data[['Mean_Latency', 'Latency_Variance', 'Total_Attempt_Volume', 'Path_Violations']]
    y = np.where(data['Concept_Accuracy'] >= 0.75, 1, 0)
    rf_engine = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    rf_engine.fit(X, y)
    return rf_engine

classification_model = train_rf_classifier(df_analytics)

# --- USER INTERFACE PRESENTATION HEADER ---
st.title("🧠 AI-Driven Technical Skill Gap Analysis & Path Routing Engine")
st.markdown("##### Continuous Academic Diagnostic Dashboard — Amity University MCA Dissertation Demonstration")
st.write("---")

# 1. Selection Control Plane (Name and Roll Number Dynamic Mapping)
control_col1, control_col2, control_col3 = st.columns(3)
with control_col1:
    selected_student_name = st.selectbox("👤 Select Student Name:", df_analytics['Student_Name'].unique())

# Fetch corresponding data subset automatically
student_profile = df_analytics[df_analytics['Student_Name'] == selected_student_name].copy()
corresponding_roll = student_profile['Roll_Number'].iloc[0]

with control_col2:
    st.text_input("📋 Student Roll Number:", value=corresponding_roll, disabled=True)

with control_col3:
    selected_role = st.selectbox("🎯 Target Professional Role Framework:", ['Data_Analyst', 'ML_Engineer'])

# 2. Live Machine Learning Inference Execution
X_live = student_profile[['Mean_Latency', 'Latency_Variance', 'Total_Attempt_Volume', 'Path_Violations']]
mastery_probabilities = classification_model.predict_proba(X_live)[:, 1]

# Calculate Skill Gap Index (SGI) based on ML predictions
student_profile['Inferred_Mastery'] = mastery_probabilities
student_profile['Calculated_SGI'] = (1.0 - student_profile['Inferred_Mastery']) * 100.0

# 3. Dynamic Real-Time Metric Display Cards
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("📊 Overall Performance Accuracy", f"{student_profile['Concept_Accuracy'].mean() * 100:.1f}%")
with metric_col2:
    mean_sgi = student_profile['Calculated_SGI'].mean()
    status_delta = "⚠️ Action Required" if mean_sgi > 40 else "✅ Safe Zone"
    st.metric("📉 Mean Skill Gap Index (SGI Scale)", f"{mean_sgi:.2f}", delta=status_delta, delta_color="inverse")
with metric_col3:
    st.metric("⚡ Response Processing Speed (Latency)", f"{student_profile['Mean_Latency'].mean():.2f} sec")

st.write("---")
st.markdown("## 📈 Advanced Analytics Matrix")

# --- ROW 1: CORE METRICS CHARTS ---
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("##### Continuous Mastery Probabilities (AI Core Output Model)")
    fig_mastery = px.bar(
        student_profile, x='Concept_Tag', y='Inferred_Mastery', 
        color='Inferred_Mastery', color_continuous_scale='Bluered_r', 
        range_y=[0, 1], labels={'Concept_Tag': 'Assessed Technical Core Concept', 'Inferred_Mastery': 'Latent Mastery Probability (0.00 - 1.00)'},
        template='plotly_dark'
    )
    st.plotly_chart(fig_mastery, use_container_width=True)

with chart_col2:
    st.markdown("##### Real-Time Skill Gap Index (SGI - Percentage Deficit)")
    fig_sgi = px.line(
        student_profile, x='Concept_Tag', y='Calculated_SGI', 
        labels={'Concept_Tag': 'Assessed Technical Core Concept', 'Calculated_SGI': 'Skill Gap Index Score (0.0 - 100.0)'},
        markers=True, template='plotly_dark'
    )
    st.plotly_chart(fig_sgi, use_container_width=True)

# --- ROW 2: ADVANCED RESEARCH CHARTS (DONUT & SCATTER DIAGRAMS) ---
chart_col3, chart_col4 = st.columns(2)
with chart_col3:
    st.markdown("##### Target Professional Role Readiness Profile Index")
    role_readiness_score = 100.0 - mean_sgi
    fig_donut = go.Figure(data=[go.Pie(
        labels=['Demonstrated Competency', 'Identified Skill Gaps Matrix'],
        values=[role_readiness_score, mean_sgi],
        hole=.5,
        marker=dict(colors=['#2E7D32', '#C62828']),
        textinfo='label+percent'
    )])
    fig_donut.update_layout(template='plotly_dark', showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_donut, use_container_width=True)

with chart_col4:
    st.markdown("##### Cognitive Telemetry Vector: Latency vs Performance Accuracy")
    fig_scatter = px.scatter(
        df_analytics, x='Mean_Latency', y='Concept_Accuracy',
        color='Concept_Tag', size='Total_Attempt_Volume',
        labels={'Mean_Latency': 'Response Latency (Seconds)', 'Concept_Accuracy': 'Observed Baseline Accuracy'},
        template='plotly_dark'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.write("---")
st.markdown("### 🎮 Control Center Interface Operations")

btn_col1, btn_col2, btn_col3 = st.columns(3)
with btn_col1:
    if st.button("🎯 Execute Personalized Path Routing", use_container_width=True):
        st.session_state.render_recommendations = True
with btn_col2:
    if st.button("📊 Show Full Telemetry Dataset", use_container_width=True):
        st.session_state.render_cohort_trends = True
with btn_col3:
    if st.button("📁 Clear Screen Nodes", use_container_width=True):
        st.session_state.render_recommendations = False
        st.session_state.render_cohort_trends = False

# 5. Smart Recommendation Knowledge Graph Output
if st.session_state.render_recommendations:
    st.write("---")
    st.markdown("#### 🎯 Prerequisite-Aware Adaptive Learning Pathways")
    
    oer_vault = {
        'Statistics': '🔗 [OpenStax Lecture Series: Statistical Analysis Theory](https://openstax.org)',
        'Linear_Algebra': '🔗 [MIT OpenCourseWare Matrix Transformations and Linear Algebra](https://ocw.mit.edu)',
        'SQL_Optimization': '🔗 [PostgreSQL Performance Tuning Documentation Manual](https://postgresql.org)',
        'Predictive_Modeling': '🔗 [Google Developers Machine Learning Crash-Course Systems](https://developers.google.com)',
        'Model_Deployment': '🔗 [Linux Environment & Cloud Architecture Blueprints](https://kernel.org)'
    }
    
    detected_gaps = student_profile[student_profile['Calculated_SGI'] >= 35.0]
    
    if not detected_gaps.empty:
        for _, row in detected_gaps.iterrows():
            failure_prob = row['Calculated_SGI']
            mastery_percentage = 100.0 - failure_prob
            
            st.error(f"🚨 **Critical Competency Deficiency Identified in Domain: {row['Concept_Tag']}** | Continuous Skill Gap Index (SGI): {row['Calculated_SGI']:.1f}")
            st.markdown(f"* **Target Remediation Open Educational Resource (OER) Link:** {oer_vault.get(row['Concept_Tag'])}")
            st.caption(f"Inference Reason: Tree-Ensemble predicts a conceptual failure probability of {failure_prob:.1f}% (Estimated Latent Mastery: {mastery_percentage:.1f}%) due to processing latency fluctuations and pattern deviations.")
    else:
        st.success("✨ Excellent! This candidate meets all performance thresholds for the target corporate role framework.")

# 6. Full Data View Pane
if st.session_state.render_cohort_trends:
    st.write("---")
    st.markdown("#### 📊 Operational Data Table Matrix (Cleaned Student Telemetry Dataset)")
    st.dataframe(df_analytics, use_container_width=True)