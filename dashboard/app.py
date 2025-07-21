# appy.py - Enhanced Movie Recommendation Dashboard
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import subprocess
import time
import requests
from io import BytesIO

# Configuration
MODEL_DIR = "models"
DATA_DIR = "data"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "e835a4cfc2ffd54fedcfc4d94f80b4fe")  # Fallback key

# Set page config
st.set_page_config(
    page_title="Movie Recommendation Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
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
        
        # Extract genres if available
        if 'genres' in data['movies'].columns:
            # Split genres and create a set of all unique genres
            all_genres = set()
            for genres in data['movies']['genres'].dropna():
                for genre in genres.split(','):
                    all_genres.add(genre.strip())
            data['all_genres'] = sorted(all_genres)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_movie_poster(poster_path):
    """Fetch movie poster from TMDB"""
    if not poster_path or pd.isna(poster_path):
        return None
    
    try:
        url = f"{POSTER_BASE_URL}{poster_path}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return BytesIO(response.content)
    except Exception:
        pass
    return None

def display_movie_details(movie):
    """Display detailed information about a movie"""
    with st.expander("🎬 Movie Details", expanded=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            # Fetch and display actual poster
            poster = get_movie_poster(movie.get('poster_path'))
            if poster:
                st.image(poster, caption=movie['title'], width=250)
            else:
                st.image("https://via.placeholder.com/300x450?text=Poster+Not+Available", 
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
            
            # Display genres if available
            if pd.notna(movie.get('genres')) and movie['genres'] != '':
                st.write("**Genres:**")
                genres = movie['genres'].split(',')
                genre_chips = " ".join([f"`{genre.strip()}`" for genre in genres])
                st.markdown(genre_chips)
            
            # Display overview
            if pd.notna(movie.get('overview')) and movie['overview'] != '':
                st.subheader("Overview")
                st.write(movie['overview'])
            else:
                st.warning("No overview available")

def trigger_data_update():
    """Trigger the data update process"""
    try:
        # Show spinner while updating
        with st.spinner("Updating movie database. This may take several minutes..."):
            result = subprocess.run(
                ["python", "scripts/data_ingestion.py"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.returncode == 0:
                st.success("Database update completed successfully!")
                st.cache_data.clear()  # Clear cache to load fresh data
                time.sleep(2)
                st.rerun()  # Refresh the app
            else:
                st.error(f"Update failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        st.error(f"Update process failed: {str(e)}")
    except Exception as e:
        st.error(f"Error during update: {str(e)}")

def main():
    # Load data
    data = load_data()
    if not data:
        st.stop()
    
    # Get movie data
    movies = data['movies']
    recommendations = data['recommendations']
    metadata = data.get('metadata', {})
    all_genres = data.get('all_genres', [])
    
    # Sidebar
    with st.sidebar:
        st.title("🎬 Movie Recommender")
        st.markdown("""
        **Discover movies similar to your favorites!**
        - Search for a movie by title
        - Filter by genre
        - View details and recommendations
        - Explore the movie database
        """)
        
        st.divider()
        
        # Manual update button
        st.subheader("Database Management")
        if st.button("🔄 Update Movie Database", use_container_width=True):
            trigger_data_update()
        st.caption("Fetches latest movies from TMDB")
        
        st.divider()
        
        # Genre filtering
        st.subheader("Filter Movies")
        selected_genres = st.multiselect(
            "Filter by genre:",
            options=all_genres,
            default=None,
            placeholder="Select genres..."
        )
        
        # Search box
        search_term = st.text_input("Search movies by title:", "")
        
        # Apply filters
        filtered_movies = movies
        if selected_genres:
            # Filter movies that have at least one of the selected genres
            filtered_movies = filtered_movies[
                filtered_movies['genres'].apply(
                    lambda g: any(genre in g for genre in selected_genres) 
                    if pd.notna(g) else False
                )
            ]
        
        if search_term:
            filtered_movies = filtered_movies[
                filtered_movies['title'].str.contains(search_term, case=False)
            ]
        
        # Pagination
        st.subheader("Movie Selection")
        page_size = 50
        total_pages = max(1, (len(filtered_movies) // page_size))
        
        if len(filtered_movies) > page_size:
            page = st.number_input(
                "Page:", 
                min_value=1, 
                max_value=total_pages, 
                value=1
            )
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(filtered_movies))
            page_movies = filtered_movies.iloc[start_idx:end_idx]
        else:
            page_movies = filtered_movies
        
        # Movie selection
        selected_title = st.selectbox(
            "Select a movie:",
            page_movies['title'],
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
            st.write(f"**Showing:** {len(page_movies)} of {len(filtered_movies)} movies")
        
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
            ax.set_xlim(0, 10)
            st.pyplot(fig)
        
        # Show top movies by rating
        if 'vote_count' in movies.columns and 'vote_average' in movies.columns:
            st.subheader("Top Rated Movies (100+ votes)")
            qualified = movies[movies['vote_count'] >= 100]
            if not qualified.empty:
                top_movies = qualified.sort_values(
                    'vote_average', ascending=False
                ).head(10)[['title', 'vote_average', 'release_date']]
                
                # Format release date
                top_movies['release_date'] = top_movies['release_date'].apply(
                    lambda x: x[:4] if pd.notna(x) else 'Unknown'
                )
                
                st.dataframe(
                    top_movies,
                    column_config={
                        "title": "Title",
                        "vote_average": st.column_config.NumberColumn(
                            "Rating", format="%.1f ⭐"
                        ),
                        "release_date": "Release Year"
                    },
                    hide_index=True,
                    use_container_width=True
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
    
    # Display recommendations in columns
    cols = st.columns(5)
    for idx, (rec_id, score) in enumerate(recs):
        rec_movie = movies[movies['id'] == rec_id].iloc[0]
        with cols[idx % 5]:
            # Display movie card
            with st.container(border=True):
                # Get actual poster
                poster = get_movie_poster(rec_movie.get('poster_path'))
                
                if poster:
                    st.image(poster, use_column_width=True)
                else:
                    st.image(
                        "https://via.placeholder.com/150x225?text=Poster+Not+Available",
                        use_column_width=True
                    )
                
                st.subheader(rec_movie['title'], help=f"ID: {rec_id}")
                
                # Display release year if available
                if pd.notna(rec_movie.get('release_date')) and rec_movie['release_date'] != '':
                    release_year = rec_movie['release_date'][:4]
                    st.caption(f"Released: {release_year}")
                
                # Display similarity score
                st.progress(float(score), text=f"Similarity: {score:.2f}")
                
                # Quick details
                if pd.notna(rec_movie.get('vote_average')):
                    st.write(f"⭐ {rec_movie['vote_average']}/10")
                
                # Quick genre display
                if pd.notna(rec_movie.get('genres')) and rec_movie['genres'] != '':
                    genres = rec_movie['genres'].split(',')[:2]
                    st.caption(", ".join([g.strip() for g in genres]))

if __name__ == "__main__":
    main()
