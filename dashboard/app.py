# appy.py - Streamlit Dashboard for Movie Recommendation System
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
MODEL_DIR = "models"
DATA_DIR = "data"

# Set page config
st.set_page_config(
    page_title="Movie Recommendation Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load movie data and models with caching"""
    data = {}
    try:
        # Load movie data
        movies_path = os.path.join(MODEL_DIR, 'movies_df.pkl')
        if os.path.exists(movies_path):
            data['movies'] = pd.read_pickle(movies_path)
        else:
            st.error("Movie data not found!")
            return None
        
        # Load recommendations
        rec_path = os.path.join(MODEL_DIR, 'recommendations.pkl')
        if os.path.exists(rec_path):
            data['recommendations'] = joblib.load(rec_path)
        else:
            st.error("Recommendation data not found!")
            return None
        
        # Load metadata
        meta_path = os.path.join(MODEL_DIR, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                data['metadata'] = json.load(f)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def display_movie_details(movie):
    """Display detailed information about a movie"""
    with st.expander("🎬 Movie Details", expanded=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            # Placeholder for movie poster (would require additional API integration)
            st.image("https://via.placeholder.com/300x450?text=Movie+Poster", 
                     caption=movie['title'], width=250)
        
        with col2:
            st.subheader(movie['title'])
            st.caption(f"ID: {movie['id']}")
            
            # Display release year if available
            if pd.notna(movie.get('release_date')) and movie['release_date'] != '':
                release_year = movie['release_date'][:4]
                st.write(f"**Release Year:** {release_year}")
            
            # Display rating information
            if pd.notna(movie.get('vote_average')):
                rating = movie['vote_average']
                st.write(f"**Rating:** {rating:.1f}/10 ⭐ ({movie.get('vote_count', 0)} votes)")
            
            # Display popularity
            if pd.notna(movie.get('popularity')):
                st.write(f"**Popularity Score:** {movie['popularity']:.1f}")
            
            # Display overview
            if pd.notna(movie.get('overview')) and movie['overview'] != '':
                st.subheader("Overview")
                st.write(movie['overview'])
            else:
                st.warning("No overview available")

def main():
    # Load data
    data = load_data()
    if not data:
        st.stop()
    
    # Get movie data
    movies = data['movies']
    recommendations = data['recommendations']
    metadata = data.get('metadata', {})
    
    # Sidebar
    with st.sidebar:
        st.title("🎬 Movie Recommender")
        st.markdown("""
        **Discover movies similar to your favorites!**
        - Search for a movie by title
        - View details and recommendations
        - Explore the movie database
        """)
        
        st.divider()
        
        # Search box
        search_term = st.text_input("Search movies by title:", "")
        filtered_movies = movies
        if search_term:
            filtered_movies = movies[
                movies['title'].str.contains(search_term, case=False)
            ]
        
        # Movie selection
        selected_title = st.selectbox(
            "Select a movie:",
            filtered_movies['title'],
            index=None,
            placeholder="Choose a movie..."
        )
        
        st.divider()
        
        # System info
        st.subheader("System Status")
        if metadata:
            trained_at = datetime.fromisoformat(metadata['trained_at'])
            st.write(f"**Last Trained:** {trained_at.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Movies in Database:** {metadata['movie_count']}")
        
        st.divider()
        st.markdown("Built with ❤️ using Streamlit")
    
    # Main content
    st.title("Movie Recommendation Dashboard")
    
    if not selected_title:
        st.info("👈 Select a movie from the sidebar to get recommendations")
        st.divider()
        
        # Show movie statistics
        st.subheader("📊 Movie Database Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Movies", len(movies))
        with col2:
            min_year = metadata.get('min_year', 'N/A')
            st.metric("Oldest Movie", min_year)
        with col3:
            max_year = metadata.get('max_year', 'N/A')
            st.metric("Newest Movie", max_year)
        
        # Show rating distribution
        if 'vote_average' in movies.columns:
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots()
            movies['vote_average'].plot.hist(
                bins=20, ax=ax, color='skyblue', edgecolor='black'
            )
            ax.set_xlabel("Rating (0-10)")
            ax.set_ylabel("Number of Movies")
            st.pyplot(fig)
        
        # Show top movies by rating
        if 'vote_count' in movies.columns and 'vote_average' in movies.columns:
            st.subheader("Top Rated Movies")
            qualified = movies[movies['vote_count'] >= 100]
            if not qualified.empty:
                top_movies = qualified.sort_values(
                    'vote_average', ascending=False
                ).head(10)[['title', 'vote_average', 'release_date']]
                st.dataframe(
                    top_movies,
                    column_config={
                        "title": "Title",
                        "vote_average": st.column_config.NumberColumn(
                            "Rating", format="%.1f ⭐"
                        ),
                        "release_date": "Release Date"
                    },
                    hide_index=True
                )
        return
    
    # Get selected movie
    selected_movie = movies[movies['title'] == selected_title].iloc[0]
    movie_id = selected_movie['id']
    
    # Display movie details
    display_movie_details(selected_movie)
    
    # Recommendations
    st.divider()
    st.subheader("🎯 Recommended Movies")
    
    if movie_id not in recommendations:
        st.warning("No recommendations available for this movie")
        return
    
    # Get top recommendations
    recs = recommendations[movie_id][:10]  # Top 10
    
    # Display recommendations
    cols = st.columns(5)
    for idx, (rec_id, score) in enumerate(recs):
        rec_movie = movies[movies['id'] == rec_id].iloc[0]
        with cols[idx % 5]:
            # Display movie card
            with st.container(border=True):
                # Placeholder for poster
                st.image(
                    "https://via.placeholder.com/150x225?text=Poster",
                    use_column_width=True
                )
                st.subheader(rec_movie['title'])
                
                # Display release year if available
                if pd.notna(rec_movie.get('release_date')) and rec_movie['release_date'] != '':
                    release_year = rec_movie['release_date'][:4]
                    st.caption(f"Released: {release_year}")
                
                # Display similarity score
                st.progress(float(score), text=f"Similarity: {score:.2f}")
                
                # Quick details
                if pd.notna(rec_movie.get('vote_average')):
                    st.write(f"⭐ {rec_movie['vote_average']}/10")

if __name__ == "__main__":
    main()
