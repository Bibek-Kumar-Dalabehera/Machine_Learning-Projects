from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movie dataset
df = pd.read_csv('movies.csv')

# Fill missing values with empty string for relevant features
selected_feature = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_feature:
    df[feature] = df[feature].fillna('')

# Combine relevant features
df['combined_features'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']

# Create TF-IDF matrix and similarity matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle movie recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    
    # Find closest match for the movie title
    list_of_all_titles = df['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_movie = df[df.title == close_match].index[0]
        similarity_score = list(enumerate(cosine_sim[index_of_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        # Display top 10 similar movies
        recommended_movies = []
        for i in range(1, 11):
            index = sorted_similar_movies[i][0]
            recommended_movies.append(df.iloc[index]['title'])
        
        return render_template('index.html', movie_name=close_match, recommended_movies=recommended_movies)
    else:
        return render_template('index.html', movie_name="Movie not found", recommended_movies=[])

if __name__ == '__main__':
    app.run(debug=True,port=3000)
