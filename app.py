"""
Title: Weather-Based Movie & Music Recommendation System
Platform: Streamlit
Author: Bharti Insan
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime

# --- APP CONFIG ---
st.set_page_config(
    page_title="Weather Entertainment Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
for key in ['show_recommendations', 'show_analytics', 'show_all_movies', 'show_playlist']:
    if key not in st.session_state:
        st.session_state[key] = False

# --- LOAD DATA ---
@st.cache_data
def load_data():
    import os
    csv_path = 'data/cleaned_movies_data.csv'
    
    # Song mappings
    songs = {
        'Romance': "Perfect - Ed Sheeran",
        'Action': "Back In Black - AC/DC",
        'Drama': "Fix You - Coldplay",
        'Thriller': "Nightcall - Kavinsky",
        'Comedy': "Happy - Pharrell Williams",
        'Sci-Fi': "Starboy - The Weeknd",
        'Horror': "Thriller - Michael Jackson"
    }
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.rename(columns={'Weather': 'Associated_Weather'}, inplace=True)
        df['Recommended_Song'] = df['Genre'].map(songs)
        return df
    else:
        st.warning("Data file not found!")
        return None

df = load_data()

if df is None or len(df) == 0:
    st.error("❌ No data available. Please run data_pipeline.py first.")
    st.stop()

# --- ML MODEL ---
@st.cache_resource
def load_model(data):
    X = data[['Budget', 'Runtime']]
    y = data['Revenue']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

ml_model = load_model(df)

# --- HEADER ---
st.title("🛰️ Weather Entertainment Analytics Platform")

# --- WEATHER INPUT ---
col_input, col_status, col_mood = st.columns(3)

# Popular Indian cities
indian_cities = [
    "New Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Indore", "Chandigarh", "Surat", "Nagpur", "Bhopal"
]

with col_input:
    city = st.selectbox("🇮🇳 Select City:", indian_cities, index=0)

# Weather to genre mapping
weather_map = {
    'Rain': ('🌧', 'Rainy', '😌 Calm', 'Romance'),
    'Drizzle': ('🌧', 'Rainy', '😌 Calm', 'Drama'),
    'Thunderstorm': ('⛈️', 'Thunderstorm', '😰 Intense', 'Horror'),
    'Clear': ('☀️', 'Sunny', '⚡ High Energy', 'Action'),
    'Sunny': ('☀️', 'Sunny', '⚡ High Energy', 'Comedy'),
    'Cloudy': ('☁️', 'Cloudy', '🧠 Focus', 'Thriller'),
    'Snow': ('❄️', 'Snowy', '❄️ Chill', 'Sci-Fi'),
}

# Get actual weather from API
API_KEY = "602d5583b38dfd7d91d95392cfec03d5"
try:
    res = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric", timeout=5).json()
    weather = res.get('weather', [{}])[0].get('main', 'Clouds')
    temp = res.get('main', {}).get('temp', 24.0)
except:
    weather, temp = "Cloudy", 24.0

# DEMO MODE: Allow manual weather override
st.write("---")
st.markdown("### 🎛️ **DEMO MODE: Change Weather to See Recommendations**")
demo_col1, demo_col2, demo_col3 = st.columns(3)

with demo_col1:
    st.markdown("**Override Weather:**")
    weather_override = st.selectbox(
        "Select Weather Type:",
        list(weather_map.keys()),
        index=list(weather_map.keys()).index(weather) if weather in weather_map else 4,
        key="weather_select"
    )
    weather = weather_override

with demo_col2:
    st.markdown("**Temperature:**")
    temp = st.slider("Select Temperature (°C):", 5.0, 45.0, temp, key="temp_slider")

with demo_col3:
    st.markdown("**Live Status:**")
    st.info(f"Weather: {weather}\nTemp: {temp}°C")

st.write("---")

# Get recommendation based on selected weather
icon, w_label, mood, genre = weather_map.get(weather, ('☁️', 'Cloudy', '🧠 Focus', 'Thriller'))

with col_status:
    st.markdown(f"**Current Weather:**\n### {icon} {w_label}\n`{temp}°C`")

with col_mood:
    st.markdown(f"**Recommended Mood:** {mood}")

st.write("---")

# --- SMART RECOMMENDATIONS ---
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### 🎬 Movies for {w_label} Weather")
    movies = df[df['Associated_Weather'] == w_label].nlargest(3, 'IMDb_Rating')
    for _, row in movies.iterrows():
        st.info(f"🍿 **{row['Title']}**\n⭐ {row['IMDb_Rating']}/10")

with col2:
    st.markdown(f"### 🎵 Songs for {w_label} Weather")
    songs_list = df[df['Associated_Weather'] == w_label].drop_duplicates('Recommended_Song')
    for _, row in songs_list.head(3).iterrows():
        st.success(f"🎧 **{row['Recommended_Song']}**")

# --- VISUALIZATIONS ---
st.write("---")
st.markdown("## 📈 Analytics")

genre_select = st.multiselect("Select Genres:", df['Genre'].unique(), default=df['Genre'].unique()[:3])
filtered = df[df['Genre'].isin(genre_select)]

viz_col1, viz_col2 = st.columns(2)
viz_col3, viz_col4 = st.columns(2)

with viz_col1:
    fig1 = px.scatter(filtered, x="Budget", y="Revenue", color="Genre", size="IMDb_Rating", template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

with viz_col2:
    fig2 = px.pie(filtered, names="Genre", values="Budget", hole=0.4, template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

with viz_col3:
    fig3 = px.histogram(df, x="Associated_Weather", y="Revenue", color="Genre", barmode="group", template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

with viz_col4:
    matrix = df.groupby(['Genre', 'Associated_Weather'])['IMDb_Rating'].mean().unstack().fillna(0)
    fig4 = px.imshow(matrix, text_auto=True, color_continuous_scale="Viridis", template="plotly_dark")
    st.plotly_chart(fig4, use_container_width=True)

# --- INTERACTIVE BUTTONS ---
st.write("---")
st.markdown("## 🎮 Control Panel")

btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

with btn_col1:
    if st.button("🎯 Recommendations", use_container_width=True):
        st.session_state.show_recommendations = not st.session_state.show_recommendations

with btn_col2:
    if st.button("📊 Analytics", use_container_width=True):
        st.session_state.show_analytics = not st.session_state.show_analytics

with btn_col3:
    if st.button("🎬 All Movies", use_container_width=True):
        st.session_state.show_all_movies = not st.session_state.show_all_movies

with btn_col4:
    if st.button("🎵 Playlist", use_container_width=True):
        st.session_state.show_playlist = not st.session_state.show_playlist

# --- RECOMMENDATIONS SECTION ---
if st.session_state.show_recommendations:
    st.markdown("### 🎯 Smart Recommendations")
    rec1, rec2 = st.columns(2)
    
    with rec1:
        rec_movies = df[df['Associated_Weather'] == w_label].nlargest(5, 'IMDb_Rating')
        for idx, (_, row) in enumerate(rec_movies.iterrows(), 1):
            st.markdown(f"**#{idx}** {row['Title']} | ⭐ {row['IMDb_Rating']}/10")
    
    with rec2:
        for idx, (_, row) in enumerate(df[df['Associated_Weather'] == w_label].drop_duplicates('Recommended_Song').head(5).iterrows(), 1):
            st.markdown(f"**#{idx}** 🎧 {row['Recommended_Song']}")

# --- ANALYTICS SECTION ---
if st.session_state.show_analytics:
    st.markdown("### 📊 Analytics Dashboard")
    
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("📽️ Movies", len(df['Title'].unique()))
    met2.metric("⭐ Avg Rating", f"{df['IMDb_Rating'].mean():.2f}/10")
    met3.metric("💰 Total Budget", f"${df['Budget'].sum()/1e9:.2f}B")
    met4.metric("💵 Total Revenue", f"${df['Revenue'].sum()/1e9:.2f}B")

# --- MOVIES SECTION ---
if st.session_state.show_all_movies:
    st.markdown("### 🍿 Movie Database")
    
    g_filter = st.multiselect("Genre:", df['Genre'].unique(), key="movies_genre")
    r_filter = st.slider("Min Rating:", 0.0, 10.0, 0.0, key="movies_rating")
    
    filtered_movies = df[(df['Genre'].isin(g_filter)) & (df['IMDb_Rating'] >= r_filter)] if g_filter else df
    st.dataframe(filtered_movies[['Title', 'Genre', 'IMDb_Rating', 'Budget', 'Revenue']].sort_values('IMDb_Rating', ascending=False))

# --- PLAYLIST SECTION ---
if st.session_state.show_playlist:
    st.markdown("### 🎧 Playlist Creator")
    
    p_genres = st.multiselect("Select Genres:", df['Genre'].unique(), key="playlist_genre")
    p_rating = st.slider("Min Rating:", 0.0, 10.0, 5.0, key="playlist_rating")
    
    if st.button("Create Playlist"):
        if p_genres:
            playlist = df[(df['Genre'].isin(p_genres)) & (df['IMDb_Rating'] >= p_rating)].drop_duplicates('Recommended_Song')
            st.success(f"✅ Playlist created with {len(playlist)} songs!")
            for idx, (_, row) in enumerate(playlist.head(10).iterrows(), 1):
                st.markdown(f"{idx}. 🎵 {row['Recommended_Song']} ({row['Genre']})")
        else:
            st.warning("Select genres first!")

# --- SIDEBAR ML ---
st.sidebar.markdown("### 🔮 Revenue Predictor")
budget_input = st.sidebar.slider("Budget (Millions):", 5.0, 300.0, 100.0)
runtime_input = st.sidebar.slider("Runtime (Minutes):", 80, 180, 120)

if st.sidebar.button("Predict Revenue"):
    pred = ml_model.predict([[budget_input * 1e6, runtime_input]])
    st.sidebar.metric("Predicted Revenue", f"${pred[0]/1e6:.2f}M")
