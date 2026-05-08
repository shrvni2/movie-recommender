```python
# models/recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib

class MovieRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.user_item_matrix = None
        self.item_features = None
        self.nmf_model = None
        self.nbrs_model = None

    def build_user_item_matrix(self):
        self.user_item_matrix = self.ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
        self.user_item_matrix.fillna(0, inplace=True)

    def build_item_features(self):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.item_features = tfidf_vectorizer.fit_transform(self.movies_df['genres'])

    def train_nmf_model(self, n_components=10):
        self.nmf_model = NMF(n_components=n_components, random_state=42)
        self.nmf_model.fit(self.user_item_matrix)

    def train_nbrs_model(self, n_neighbors=10):
        self.nbrs_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
        self.nbrs_model.fit(self.item_features)

    def get_recommendations(self, user_id, num_recommendations=10):
        user_vector = self.nmf_model.transform(self.user_item_matrix.loc[[user_id]])
        scores = cosine_similarity(user_vector, self.item_features)
        top_indices = np.argsort(-scores[0])[:num_recommendations]
        return self.movies_df.iloc[top_indices]

    def get_similar_movies(self, movie_id, num_similar=10):
        movie_vector = self.item_features[movie_id]
        scores = cosine_similarity(movie_vector.reshape(1, -1), self.item_features)
        top_indices = np.argsort(-scores[0])[:num_similar]
        return self.movies_df.iloc[top_indices]

    def save_models(self):
        joblib.dump(self.nmf_model, 'nmf_model.joblib')
        joblib.dump(self.nbrs_model, 'nbrs_model.joblib')

    def load_models(self):
        self.nmf_model = joblib.load('nmf_model.joblib')
        self.nbrs_model = joblib.load('nbrs_model.joblib')

def main():
    movies_df = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
    recommender = MovieRecommender(movies_df, ratings_df)
    recommender.build_user_item_matrix()
    recommender.build_item_features()
    recommender.train_nmf_model()
    recommender.train_nbrs_model()
    recommender.save_models()

if __name__ == '__main__':
    main()
```