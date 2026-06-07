"""
Title: AI-Driven Cinematic Weather & Mood Dashboard Engine
Author: Bharti Insan (Amity University Digital Campus Project Portfolio)
Deployment Platform: Streamlit Cloud Only
Architecture Sync Format: Modular Directory Alignment
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

# --- INITIAL APP ENVIRONMENT SETUP ---
st.set_page_config(
	page_title="Weather Entertainment Platform",
	page_icon="🎬",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Initialize session state for interactive buttons
if 'show_recommendations' not in st.session_state:
	st.session_state.show_recommendations = False
if 'show_analytics' not in st.session_state:
	st.session_state.show_analytics = False
if 'show_all_movies' not in st.session_state:
	st.session_state.show_all_movies = False
if 'show_playlist' not in st.session_state:
	st.session_state.show_playlist = False

# --- DETERMINISTIC MOVIE & SONG DATA MATRIX ---
@st.cache_data
def load_production_dataset():
	genres = ['Romance', 'Action', 'Drama', 'Thriller', 'Comedy', 'Sci-Fi', 'Horror']
	movie_manifest = {
		'Romance': ["Titanic", "The Notebook", "Before Sunrise", "La La Land", "About Time"],
		'Action': ["The Dark Knight", "Mad Max: Fury Road", "Inception", "Gladiator", "John Wick"],
		'Drama': ["The Shawshank Redemption", "Forrest Gump", "The Godfather", "Whiplash", "The Green Mile"],
		'Thriller': ["Se7en", "Zodiac", "Gone Girl", "Shutter Island", "Prisoners"],
		'Comedy': ["The Hangover", "Superbad", "Step Brothers", "Anchorman", "Office Space"],
		'Sci-Fi': ["Interstellar", "Blade Runner 2048", "Arrival", "The Matrix", "Dune"],
		'Horror': ["The Conjuring", "Hereditary", "Get Out", "It", "A Quiet Place"]
	}
	song_manifest = {
		'Romance': ["Perfect - Ed Sheeran", "Let Her Go - Passenger", "Night Changes - One Direction"],
		'Action': ["Back In Black - AC/DC", "Lose Yourself - Eminem", "Can't Stop - Red Hot Chili Peppers"],
		'Drama': ["Fix You - Coldplay", "Someone Like You - Adele", "Chasing Cars - Snow Patrol"],
		'Thriller': ["Nightcall - Kavinsky", "Bury A Friend - Billie Eilish", "Stargazing - The Neighbourhood"],
		'Comedy': ["Happy - Pharrell Williams", "Don't Stop Me Now - Queen", "Uptown Funk - Bruno Mars"],
		'Sci-Fi': ["Starboy - The Weeknd", "Midnight City - M83", "Intro - The xx"],
		'Horror': ["Thriller - Michael Jackson", "Psycho Killer - Talking Heads", "Bad Moon Rising - CCR"]
	}

	dataset = []
	cursor = 0
	for g in genres:
		m_list, s_list = movie_manifest[g], song_manifest[g]
		for i in range(15):
			title = m_list[i % len(m_list)] + f" ({2002 + (i * 2)})"
			song = s_list[i % len(s_list)]
			budget = int((25 + (cursor * 1.7)) * 1000000)
			runtime = int(90 + (cursor % 3) * 15 + (cursor * 0.2))
			rating = round(5.2 + (cursor % 4) * 0.9, 1)
			revenue = budget * (1.1 + (rating * 0.35) + np.sin(cursor) * 0.4)
			weather = "Rainy" if g in ['Romance', 'Drama'] else ("Sunny" if g in ['Action', 'Comedy'] else "Cloudy")

			dataset.append({
				"Title": title, "Genre": g, "Budget": budget, "Runtime": runtime,
				"IMDb_Rating": rating, "Revenue": revenue, "Associated_Weather": weather, "Recommended_Song": song
			})
			cursor += 1
	return pd.DataFrame(dataset)

df = load_production_dataset()

# Remove any accidental duplicate asset rows (by Title + Genre)
df = df.drop_duplicates(subset=['Title', 'Genre']).reset_index(drop=True)

# --- STREAMLIT-OPTIMIZED IN-MEMORY ML PIPELINE ---
@st.cache_resource
def load_ml_predictor(data):
	X = data[['Budget', 'Runtime']]
	y = data['Revenue']
	compiled_model = RandomForestRegressor(n_estimators=100, random_state=42)
	compiled_model.fit(X, y)
	return compiled_model

ml_predictor = load_ml_predictor(df)

# ==============================================================================
# VISUAL USER INTERFACE DISPLAY LAYER
# ==============================================================================
if os.path.exists("templates/index.html"):
	with open("templates/index.html", "r", encoding="utf-8") as html_file:
		st.markdown(html_file.read(), unsafe_allow_html=True)
else:
	st.title("🛰️ Weather Entertainment Analytics Platform")

# Dynamic Search Row Layout
col_input, col_status, col_mood = st.columns(3)

with col_input:
	city_query = st.text_input("📍 Local Observation Target:", "New Delhi")

# Integrated OpenWeatherMap Fallback Fetch
API_KEY = "602d5583b38dfd7d91d95392cfec03d5"
# Use the correct OpenWeatherMap API endpoint for current weather
api_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_query}&appid={API_KEY}&units=metric"

try:
	api_res = requests.get(api_url, timeout=5).json()
	condition_vector = api_res.get('weather', [{}])[0].get('main', 'Clouds')
	temp_vector = api_res.get('main', {}).get('temp', 24.0)
except Exception:
	condition_vector, temp_vector = "Rain", 24.0

# Context Rule Classifier
if condition_vector in ["Rain", "Drizzle", "Thunderstorm"]:
	icon, weather_lbl, mood_lbl, target_genre = "🌧", "Rainy", "😌 Calm Mood Matrix", "Romance"
elif condition_vector in ["Clear", "Sunny"]:
	icon, weather_lbl, mood_lbl, target_genre = "☀️", "Sunny", "⚡ High Energy Matrix", "Action"
else:
	icon, weather_lbl, mood_lbl, target_genre = "☁️", "Cloudy", "🧠 High Focus Matrix", "Thriller"

with col_status:
	st.markdown(f"**Current Node Condition:**\n### {icon} {weather_lbl}\n`Telemetry: {temp_vector}°C`")

with col_mood:
	st.markdown(f"**Calculated Focus Vector:**\n### {mood_lbl}")

st.write("---")

# Smart Recommendation Content Modules
col_m_view, col_s_view = st.columns(2)
target_assets = df[df['Genre'] == target_genre].drop_duplicates(subset=['Title']).head(3)

with col_m_view:
	st.markdown("### 🎬 Contextual Content Assets")
	for movie in target_assets['Title']:
		st.info(f"🍿 **{movie}**")

with col_s_view:
	st.markdown("### 🎵 Core Playlist Track Suggestions")
	for song in target_assets['Recommended_Song']:
		st.success(f"🎧 **{song}**")

# ==============================================================================
# DYNAMIC PLOTLY INTERACTIVE VISUALIZATIONS SECTION
# ==============================================================================
st.write("---")
st.markdown("## 📈 Interactive System Visualization Modules")

genre_selections = st.multiselect("Isolate Data Dimensions:", options=df['Genre'].unique().tolist(), default=df['Genre'].unique().tolist()[:4])
filtered_analytics_space = df[df['Genre'].isin(genre_selections)]

graph_row1_L, graph_row1_R = st.columns(2)
graph_row2_L, graph_row2_R = st.columns(2)

with graph_row1_L:
	st.markdown("#### `[Revenue Chart Analysis]`")
	fig1 = px.scatter(filtered_analytics_space, x="Budget", y="Revenue", color="Genre", size="IMDb_Rating", template="plotly_dark")
	st.plotly_chart(fig1, use_container_width=True)

with graph_row1_R:
	st.markdown("#### `[Genre Allocation Index]`")
	fig2 = px.pie(filtered_analytics_space, names="Genre", values="Budget", hole=0.4, template="plotly_dark")
	st.plotly_chart(fig2, use_container_width=True)

with graph_row2_L:
	st.markdown("#### `[Weather Yield Variations]`")
	fig3 = px.histogram(df, x="Associated_Weather", y="Revenue", color="Genre", barmode="group", template="plotly_dark")
	st.plotly_chart(fig3, use_container_width=True)

with graph_row2_R:
	st.markdown("#### `[Mood Correlation Space]`")
	matrix_pivot = df.groupby(['Genre', 'Associated_Weather'])['IMDb_Rating'].mean().unstack().fillna(0)
	fig4 = px.imshow(matrix_pivot, text_auto=True, color_continuous_scale="Viridis", template="plotly_dark")
	st.plotly_chart(fig4, use_container_width=True)

# Machine Learning Predictive Control Center Sidebar
st.sidebar.markdown("### 🔮 Advanced Predictive Inference")
sidebar_b = st.sidebar.slider("Asset Budget ($ Millions)", 5.0, 250.0, 75.0)
sidebar_r = st.sidebar.slider("Runtime Duration (Mins)", 75, 195, 120)

if st.sidebar.button("Compute Forecast Models"):
	inference_return = ml_predictor.predict(np.array([[sidebar_b * 1000000, sidebar_r]]))
	# `predict` returns an array; format the first value
	st.sidebar.metric("Predicted Worldwide Returns", f"${inference_return[0]/1000000:.2f} Million")

# ==============================================================================
# ENHANCED INTERACTIVE FEATURES SECTION
# ==============================================================================
st.write("---")
st.markdown("## 🎮 Interactive Control Panel")

control_col1, control_col2, control_col3, control_col4 = st.columns(4)

with control_col1:
	if st.button("🎯 Get Smart Recommendations", use_container_width=True):
		st.session_state.show_recommendations = True
		st.success(f"✅ Recommendations based on {weather_lbl} weather!")

with control_col2:
	if st.button("📊 Show Detailed Analytics", use_container_width=True):
		st.session_state.show_analytics = True
		st.info("📈 Analytics panel activated!")

with control_col3:
	if st.button("🎬 Browse All Movies", use_container_width=True):
		st.session_state.show_all_movies = True
		st.info("🍿 Movie browser opened!")

with control_col4:
	if st.button("🎵 Create Playlist", use_container_width=True):
		st.session_state.show_playlist = True
		st.info("🎧 Playlist generator ready!")

# Weather-Based Smart Recommendations Section
if st.session_state.get('show_recommendations', False):
	st.markdown("### 🎯 Smart Weather-Based Recommendations")
	
	rec_col1, rec_col2 = st.columns(2)
	
	with rec_col1:
		st.markdown(f"#### 🎬 Top Movies for {weather_lbl} Weather")
		rec_movies = df[df['Associated_Weather'] == weather_lbl].nlargest(5, 'IMDb_Rating')
		
		for idx, row in rec_movies.iterrows():
			col_rank, col_info = st.columns([0.5, 4])
			with col_rank:
				st.markdown(f"**#{idx+1}**")
			with col_info:
				progress_val = row['IMDb_Rating'] / 10
				st.markdown(f"**{row['Title']}** ({row['Genre']})")
				st.progress(progress_val)
				st.caption(f"⭐ {row['IMDb_Rating']}/10 | Runtime: {int(row['Runtime'])} min")
	
	with rec_col2:
		st.markdown(f"#### 🎵 Top Songs for {weather_lbl} Weather")
		rec_songs = df[df['Associated_Weather'] == weather_lbl].nlargest(5, 'IMDb_Rating')
		
		for idx, row in rec_songs.iterrows():
			col_rank, col_info = st.columns([0.5, 4])
			with col_rank:
				st.markdown(f"**#{idx+1}**")
			with col_info:
				st.markdown(f"🎧 **{row['Recommended_Song']**")
				st.caption(f"From {row['Genre']} genre | Rating: ⭐ {row['IMDb_Rating']}/10")

# Detailed Analytics Section
if st.session_state.get('show_analytics', False):
	st.markdown("### 📊 Detailed Analytics Dashboard")
	
	# Create detailed metrics
	analytics_col1, analytics_col2, analytics_col3, analytics_col4 = st.columns(4)
	
	with analytics_col1:
		total_movies = len(df['Title'].unique())
		st.metric("📽️ Total Movies", total_movies)
	
	with analytics_col2:
		avg_rating = df['IMDb_Rating'].mean()
		st.metric("⭐ Avg Rating", f"{avg_rating:.2f}/10")
	
	with analytics_col3:
		total_budget = df['Budget'].sum()
		st.metric("💰 Total Budget", f"${total_budget/1e9:.2f}B")
	
	with analytics_col4:
		total_revenue = df['Revenue'].sum()
		st.metric("💵 Total Revenue", f"${total_revenue/1e9:.2f}B")
	
	# ROI Analysis
	st.markdown("#### 🎯 ROI Analysis by Genre")
	roi_data = df.groupby('Genre').apply(lambda x: ((x['Revenue'].sum() - x['Budget'].sum()) / x['Budget'].sum() * 100)).reset_index()
	roi_data.columns = ['Genre', 'ROI %']
	fig_roi = px.bar(roi_data, x='Genre', y='ROI %', color='ROI %', color_continuous_scale='RdYlGn', template='plotly_dark')
	st.plotly_chart(fig_roi, use_container_width=True)

# All Movies Browser
if st.session_state.get('show_all_movies', False):
	st.markdown("### 🍿 Complete Movie Database")
	
	# Filter options
	filter_col1, filter_col2 = st.columns(2)
	
	with filter_col1:
		genre_filter = st.multiselect("Filter by Genre:", df['Genre'].unique(), default=df['Genre'].unique())
	
	with filter_col2:
		rating_filter = st.slider("Minimum Rating:", 0.0, 10.0, 0.0)
	
	# Apply filters
	filtered_movies = df[(df['Genre'].isin(genre_filter)) & (df['IMDb_Rating'] >= rating_filter)]
	
	# Display as table with sorting
	sort_col = st.selectbox("Sort by:", ['IMDb_Rating', 'Budget', 'Revenue', 'Runtime'])
	filtered_movies = filtered_movies.sort_values(sort_col, ascending=False)
	
	st.dataframe(
		filtered_movies[['Title', 'Genre', 'IMDb_Rating', 'Budget', 'Revenue', 'Associated_Weather']].head(20),
		use_container_width=True
	)

# Playlist Creator
if st.session_state.get('show_playlist', False):
	st.markdown("### 🎧 Custom Playlist Creator")
	
	playlist_col1, playlist_col2 = st.columns(2)
	
	with playlist_col1:
		selected_genres = st.multiselect("Select genres for playlist:", df['Genre'].unique())
		min_rating = st.slider("Minimum song rating:", 0.0, 10.0, 5.0)
	
	with playlist_col2:
		if st.button("Generate Playlist", use_container_width=True):
			playlist_songs = df[(df['Genre'].isin(selected_genres)) & (df['IMDb_Rating'] >= min_rating)]
			st.success(f"✅ Created playlist with {len(playlist_songs)} songs!")
			
			st.markdown("#### Your Playlist:")
			for idx, (_, row) in enumerate(playlist_songs.head(10).iterrows(), 1):
				st.markdown(f"{idx}. 🎵 **{row['Recommended_Song']}** - {row['Genre']} ({row['IMDb_Rating']}⭐)")

# Weather Statistics
st.write("---")
st.markdown("## 🌤️ Weather Performance Insights")

weather_stats_col1, weather_stats_col2 = st.columns(2)

with weather_stats_col1:
	# Average Revenue by Weather
	weather_revenue = df.groupby('Associated_Weather')['Revenue'].mean()
	fig_weather = px.bar(
		x=weather_revenue.index, 
		y=weather_revenue.values,
		labels={'x': 'Weather', 'y': 'Avg Revenue'},
		title="Average Revenue by Weather",
		template="plotly_dark"
	)
	st.plotly_chart(fig_weather, use_container_width=True)

with weather_stats_col2:
	# Genre distribution
	genre_count = df['Genre'].value_counts()
	fig_genre = px.pie(
		names=genre_count.index,
		values=genre_count.values,
		title="Genre Distribution",
		template="plotly_dark"
	)
	st.plotly_chart(fig_genre, use_container_width=True)