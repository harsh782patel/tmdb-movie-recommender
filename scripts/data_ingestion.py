import schedule
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import pandas as pd
import duckdb
import os
import json
import re
import logging
import argparse
from datetime import datetime

# Configuration
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'ingestion.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_ingestion')

# Get API key from environment variable with fallback
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "e835a4cfc2ffd54fedcfc4d94f80b4fe")

def create_session_with_retries():
    """Create requests session with retry strategy"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=10,
        backoff_factor=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20
    )
    session.mount("https://", adapter)
    
    proxies = {}
    if os.getenv('HTTP_PROXY'):
        proxies['http'] = os.getenv('HTTP_PROXY')
    if os.getenv('HTTPS_PROXY'):
        proxies['https'] = os.getenv('HTTPS_PROXY')
    if proxies:
        session.proxies = proxies
    
    return session

def fetch_tmdb_data(pages=50):
    """Fetch multiple pages of movie data from TMDB API"""
    base_url = "https://api.themoviedb.org/3/movie/popular"
    headers = {
        "Accept": "application/json",
        "User-Agent": "MovieRecommender/1.0",
        "RateLimit-Limit": "40",
        "RateLimit-Remaining": "39",
        "RateLimit-Reset": "10"
    }
    
    all_movies = []
    total_movies = 0
    session = None
    
    try:
        session = create_session_with_retries()
        logger.info(f"Fetching {pages} pages from TMDB API...")
        
        # First fetch genre mappings
        genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}"
        genre_response = session.get(genre_url)
        genre_data = genre_response.json()
        genre_map = {g['id']: g['name'] for g in genre_data.get('genres', [])}
        
        for page in range(1, pages + 1):
            logger.info(f"Fetching page {page}/{pages}...")
            params = {
                'api_key': TMDB_API_KEY,
                'page': page
            }
            
            try:
                timeout = 10 + (page % 3)
                response = session.get(base_url, headers=headers, params=params, timeout=timeout)
                logger.info(f"  Page {page} status: {response.status_code}, time: {response.elapsed.total_seconds():.2f}s")
                response.raise_for_status()
                
                data = response.json()
                
                if 'results' not in data:
                    logger.warning(f"Warning: Unexpected response format on page {page}")
                    continue
                    
                # Process each movie to add genres and keep poster_path
                page_movies = []
                for movie in data['results']:
                    movie['genres'] = ', '.join([genre_map[gid] for gid in movie.get('genre_ids', []) 
                                              if gid in genre_map])
                    movie['poster_path'] = movie.get('poster_path', '')
                    page_movies.append(movie)
                
                movies_count = len(page_movies)
                all_movies.extend(page_movies)
                total_movies += movies_count
                logger.info(f"  Added {movies_count} movies from page {page}")
                
                delay = 0.5 + (page % 10)/10
                if page < pages:
                    time.sleep(delay)
                    
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"Warning: Failed for page {page}: {str(e)}")
                time.sleep(2)
                continue
                
        logger.info(f"\nSuccessfully fetched {total_movies} movies from {pages} pages")
        return pd.DataFrame(all_movies)
        
    except Exception as e:
        logger.error(f"Warning: Unexpected error: {str(e)}")
        return pd.DataFrame(all_movies)
    finally:
        if session:
            session.close()

def load_sample_data():
    """Load sample data from file when API fails"""
    sample_path = os.path.join(DATA_DIR, 'sample_movies.json')
    try:
        logger.info(f"Loading sample data from {sample_path}")
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        logger.warning("Sample data file not found")
        return pd.DataFrame()
    except json.JSONDecodeError:
        logger.warning("Error decoding sample data")
        return pd.DataFrame()

def clean_movie_data(df):
    """Clean and prepare movie data for storage"""
    if df.empty:
        return df
        
    # 1. Remove duplicate IDs
    initial_count = len(df)
    df = df.drop_duplicates(subset='id', keep='first')
    if initial_count != len(df):
        logger.info(f"Removed {initial_count - len(df)} duplicate movie records")
    
    # 2. Clean release_date format
    if 'release_date' in df.columns:
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        valid_dates = df['release_date'].apply(
            lambda x: bool(re.match(pattern, str(x))) if pd.notnull(x) else False
        )
        invalid_count = (~valid_dates).sum()
        if invalid_count:
            logger.info(f"Found {invalid_count} invalid date formats - cleaning...")
            df.loc[~valid_dates, 'release_date'] = None
        current_year = datetime.now().year
        future_dates = df['release_date'].apply(
            lambda x: int(x[:4]) > current_year + 5 if pd.notnull(x) and re.match(pattern, x) else False
        )
        if future_dates.sum() > 0:
            logger.info(f"Found {future_dates.sum()} future dates - cleaning...")
            df.loc[future_dates, 'release_date'] = None

    # 3. Fill missing critical fields
    critical_fields = ['title', 'overview', 'genres']
    for field in critical_fields:
        if field in df.columns:
            missing = df[field].isna().sum()
            if missing:
                logger.info(f"Filling {missing} missing values in {field}")
                if field == 'genres':
                    df.loc[:, field] = df[field].fillna('Unknown')
                else:
                    df.loc[:, field] = df[field].fillna('')
    
    # 4. Ensure numeric columns are properly typed
    numeric_fields = ['vote_count', 'vote_average', 'popularity']
    for field in numeric_fields:
        if field in df.columns:
            df.loc[:, field] = pd.to_numeric(df[field], errors='coerce')
            nan_count = df[field].isna().sum()
            if nan_count:
                logger.info(f"Filling {nan_count} NaN values in {field}")
                df.loc[:, field] = df[field].fillna(0)
    
    # 5. Ensure poster_path is string
    if 'poster_path' in df.columns:
        df['poster_path'] = df['poster_path'].fillna('').astype(str)
    
    return df

def store_data(df):
    """Store data in DuckDB"""
    if df.empty:
        logger.warning("No data to store. Using sample data instead.")
        df = load_sample_data()
        if df.empty:
            logger.error("No data available. Exiting.")
            return False
        else:
            logger.info(f"Using sample dataset with {len(df)} movies")
            
    df = clean_movie_data(df)
    
    # DuckDB Storage
    duckdb_path = os.path.join(DATA_DIR, 'movies.duckdb')
    duckdb_conn = None
    try:
        duckdb_conn = duckdb.connect(duckdb_path)
        duckdb_conn.execute("CREATE OR REPLACE TABLE movies AS SELECT * FROM df")
        logger.info(f"Data stored in DuckDB: {duckdb_path}")
        return True
    except duckdb.Error as e:
        logger.error(f"DuckDB error: {str(e)}")
        return False
    finally:
        if duckdb_conn:
            duckdb_conn.close()

def create_sample_file():
    """Create sample data file for fallback"""
    sample_data = [
        {
            "id": 1,
            "title": "Sample Movie",
            "overview": "This is a sample movie",
            "release_date": "2023-01-01",
            "vote_average": 7.5,
            "vote_count": 100,
            "popularity": 50.0,
            "genres": "Action, Adventure",
            "poster_path": "/sample1.jpg"
        },
        {
            "id": 2,
            "title": "Another Sample",
            "overview": "Another sample movie",
            "release_date": "2023-02-15",
            "vote_average": 8.0,
            "vote_count": 150,
            "popularity": 75.0,
            "genres": "Drama, Romance",
            "poster_path": "/sample2.jpg"
        }
    ]
    sample_path = os.path.join(DATA_DIR, 'sample_movies.json')
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    logger.info(f"Created sample data file: {sample_path}")

def periodic_data_refresh():
    """Refresh data daily and on-demand"""
    logger.info("\nStarting periodic data refresh...")
    try:
        movies_df = fetch_tmdb_data(pages=50)
        if not movies_df.empty:
            store_data(movies_df)
            logger.info("Data refresh successful!")
            return True
        return False
    except Exception as e:
        logger.error(f"Refresh failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Movie Data Ingestion Script')
    parser.add_argument('--daemon', action='store_true', 
                        help='Run in daemon mode with scheduled refreshes')
    args = parser.parse_args()

    if os.name == 'nt':
        os.system('chcp 65001 > nul')
    
    sample_path = os.path.join(DATA_DIR, 'sample_movies.json')
    if not os.path.exists(sample_path):
        create_sample_file()
    
    try:
        movies_df = fetch_tmdb_data(pages=50) 
    except Exception as e:
        logger.error(f"Critical error during fetch: {str(e)}")
        movies_df = pd.DataFrame()
    
    if not movies_df.empty:
        store_data(movies_df)
        logger.info("Initial data stored successfully!")
    
    if not args.daemon:
        logger.info("Initial data ingestion completed. Exiting.")
        exit(0)
    
    schedule.every().day.at("03:00").do(periodic_data_refresh)
    
    logger.info("Running in daemon mode. Scheduled refreshes set for 3 AM daily.")
    logger.info("Press Ctrl+C to exit...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)