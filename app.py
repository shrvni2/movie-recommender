```python
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pickle
import json

app = Flask(__name__)

# Load data
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the movie descriptions and transform them into vectors
tfidf_matrix = vectorizer.fit_transform(movies['description'])

# Calculate the cosine similarity between the vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a dictionary to map movie IDs to their indices
indices = pd.Series(movies.index, index=movies['id']).drop_duplicates()

# Function to get movie recommendations based on content-based filtering
def get_recommendations(id):
    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Function to get movie recommendations based on collaborative filtering
def get_collaborative_recommendations(user_id):
    user_ratings = ratings[ratings['userId'] == user_id]
    user_ratings = user_ratings.merge(movies, on='movieId')
    user_ratings = user_ratings.sort_values(by='rating', ascending=False)
    return user_ratings.head(10)

# Function to get hybrid movie recommendations
def get_hybrid_recommendations(user_id):
    collaborative_recommendations = get_collaborative_recommendations(user_id)
    content_based_recommendations = get_recommendations(collaborative_recommendations.iloc[0]['movieId'])
    return pd.concat([collaborative_recommendations, content_based_recommendations])

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_id = request.form['user_id']
    recommendations = get_hybrid_recommendations(int(user_id))
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/movies')
def movies_list():
    return render_template('movies.html', movies=movies)

if __name__ == '__main__':
    app.run(debug=True)
```