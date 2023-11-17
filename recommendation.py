from flask import Flask, request, jsonify
import pandas as pd
import bz2
import pickle
from sklearn.metrics.pairwise import linear_kernel

from joblib import load


# Load the saved content-based recommendation model
hash_vectorizer = load('hash_vectorizer.joblib')
cosine_sim = load('cosine_sim.joblib')
df = load('restaurant_data.joblib')
hash_matrix = load('hash_matrix.joblib')


# # Load the HashingVectorizer and cosine_sim from pickle files
# with open('hash_vectorizer.pkl', 'rb') as f:
#     hash_vectorizer = pickle.load(f)

# def decompress_pickle(file):
#     data = bz2.BZ2File(file, 'rb')
#     data = pickle.load(data)
#     return data

# # Use the function to load your cosine_sim object
# cosine_sim = decompress_pickle('cosine_sim_compressed.pbz2')

# with open('cosine_sim.pkl', 'rb') as f:
#     cosine_sim = pickle.load(f)

#df = pd.read_excel('zomato.xlsx')

# Feature selection: Select the relevant columns for content-based filtering
features = ['name', 'location', 'rest_type', 'cuisines', 'dish_liked', 'rate', 'city']
df['content'] = df[features].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# # Transform using HashingVectorizer
# hash_matrix = hash_vectorizer.transform(df['content'])

# Define the recommend function
def recommend(user_input):
    # Transform user input using HashingVectorizer
    user_hash = hash_vectorizer.transform([user_input])

    # Compute the cosine similarity between user input and restaurants
    cosine_sim_user = linear_kernel(user_hash, hash_matrix).flatten()

    # Get indices of restaurants sorted by similarity
    restaurant_indices = cosine_sim_user.argsort()[::-1]

    # Exclude the first index (self) and take top recommendations
    top_recommendations = restaurant_indices[1:6]

    # Return the top recommendations as JSON
    recommended_restaurants = df.iloc[top_recommendations]
    return recommended_restaurants[['name', 'rate', 'location', 'rest_type', 'dish_liked', 'cuisines', 'cost']]