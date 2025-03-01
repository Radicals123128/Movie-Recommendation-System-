from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import ast

app = Flask(__name__)

def load_and_process_data():
    try:
        credits_df = pd.read_csv("tmdb_5000_credits.csv.gz", compression="gzip")
        movies_df = pd.read_csv("tmdb_5000_movies.csv.gz", compression="gzip")
        # Merge the datasets on movie_id
        credits_df.columns = ['movie_id', 'title', 'cast', 'crew']
        movies_df = movies_df.rename(columns={'id': 'movie_id'})
        df = movies_df.merge(credits_df, on='movie_id')

        # Convert string representations of lists/dicts to Python objects
        df['cast'] = df['cast'].apply(ast.literal_eval)
        df['crew'] = df['crew'].apply(ast.literal_eval)
        df['genres'] = df['genres'].apply(ast.literal_eval)

        # Extract top 3 actors and director
        def get_top_3_actors(cast):
            actors = []
            for person in cast[:3]:  # Get first 3 cast members
                actors.append(person['name'])
            return ', '.join(actors)

        def get_director(crew):
            for person in crew:
                if person['job'] == 'Director':
                    return person['name']
            return ''

        # Add new columns
        df['top_actors'] = df['cast'].apply(get_top_3_actors)
        df['director'] = df['crew'].apply(get_director)
        df['genres_list'] = df['genres'].apply(lambda x: ', '.join([genre['name'] for genre in x]))

        # Create combined text for better recommendations
        df['combined_text'] = (
            df['overview'].fillna('') + ' ' +
            df['genres_list'].fillna('') + ' ' +
            df['top_actors'].fillna('') + ' ' +
            df['director'].fillna('')
        ).str.lower()

        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )

        # Create TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(df['combined_text'])

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        return df, cosine_sim

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Load the data
df, cosine_sim = load_and_process_data()

def format_currency(value):
    if pd.isna(value) or value == 0:
        return "N/A"
    return f"${value:,.2f}"

def get_recommendations(movie_name):
    try:
        if df is None or cosine_sim is None:
            return ["Error: Dataset not properly loaded. Please check the data files."]

        # Convert input to lowercase for matching
        movie_name_lower = movie_name.lower()

        # Find the movie in the dataset (case-insensitive partial match)
        matching_movies = df[df['title_x'].str.lower().str.contains(movie_name_lower, na=False)]

        if matching_movies.empty:
            # Return some popular movies as suggestions
            popular_movies = df.nlargest(5, 'popularity')
            return [f"Movie '{movie_name}' not found. Here are some popular movies:"] + \
                   [f"- {title} ({year})" for title, year in 
                    zip(popular_movies['title_x'], pd.to_datetime(popular_movies['release_date']).dt.year)]

        # Get the index of the movie
        movie_idx = matching_movies.index[0]
        movie_title = matching_movies['title_x'].iloc[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top similar movies (excluding itself)
        sim_scores = sim_scores[1:6]

        # Format recommendations with more details
        recommendations = [f"Based on '{movie_title}', here are similar movies:"]

        for i, (idx, score) in enumerate(sim_scores, 1):
            movie = df.iloc[idx]
            release_year = pd.to_datetime(movie['release_date']).year if pd.notna(movie['release_date']) else "N/A"
            
            recommendations.append(f"\n{i}. {movie['title_x']} ({release_year})")
            recommendations.append(f"   Director: {movie['director']}")
            recommendations.append(f"   Starring: {movie['top_actors']}")
            recommendations.append(f"   Genres: {movie['genres_list']}")
            recommendations.append(f"   Rating: {movie['vote_average']}/10 ({movie['vote_count']} votes)")
            recommendations.append(f"   Popularity Score: {movie['popularity']:.1f}")
            recommendations.append(f"   Budget: {format_currency(movie['budget'])}")
            recommendations.append(f"   Revenue: {format_currency(movie['revenue'])}")
            recommendations.append(f"   Similarity Score: {score:.1%}")
            recommendations.append(f"   Overview: {movie['overview']}")
            recommendations.append("")  # Empty line for spacing

        return recommendations

    except Exception as e:
        print(f"Error in recommendations: {e}")
        return [f"An error occurred while getting recommendations: {str(e)}"]

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        movie_name = request.form.get("movie")
        if movie_name:
            recommendations = get_recommendations(movie_name)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
