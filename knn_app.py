import spacy
from spacy import displacy
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from joblib import load
import streamlit as st
import requests
import os
import subprocess

def custom_tokenizer(text):
    return re.split(r'[;,]', text) #Creates a tokenizer for the vectorizer

# Helper function to download files from GitHub
def download_file(url, local_filename):
    response = requests.get(url)
    response.raise_for_status()  # Check for request errors
    with open(local_filename, 'wb') as f:
        f.write(response.content)

# URLs for the joblib and CSV files
data_url = 'https://raw.githubusercontent.com/YottaParasec/BookRecommender/main/KNN_book_data.csv'
vectorizer_url = 'https://raw.githubusercontent.com/YottaParasec/BookRecommender/main/vectorizer.joblib'
knn_url = 'https://raw.githubusercontent.com/YottaParasec/BookRecommender/main/knn_recommender.joblib'

# Download files if they don't exist
if not os.path.exists('vectorizer.joblib'):
    download_file(vectorizer_url, 'vectorizer.joblib')

if not os.path.exists('knn_recommender.joblib'):
    download_file(knn_url, 'knn_recommender.joblib')

if not os.path.exists('KNN_book_data.csv'):
    download_file(data_url, 'KNN_book_data.csv')

# Load the files after downloading
vectorizer = load('vectorizer.joblib')  # Loads a trained CountVectorizer
knn = load('knn_recommender.joblib')  # Loads a trained KNN model
df = pd.read_csv('KNN_book_data.csv')  # Loads cleaned, vectorized, and scaled book data

model_name = 'en_core_web_sm'
model_dir = spacy.util.get_package_path(model_name) if spacy.util.get_package_path else None

if not model_dir:
    print(f"{model_name} not found. Downloading...")
    subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)

# Load the SpaCy model
nlp = spacy.load(model_name)

vocabulary = set(vectorizer.vocabulary_.keys()) #Sets a variable containing a list of genres

def recommender(user_input_, neighbors=10, recommendation_type_value=1):
    doc_user = nlp(user_input_)  # Process the input using the NLP model
    user_genres = []  # Initialize an empty list to store the processed genres

    conversion_dict = {  # A dictionary to convert specific words/phrases to standard genres
        'scifi': 'science fiction',
        'sci-fi': 'science fiction',
        'sci fi': 'science fiction',
        'comic': 'comic book',
        'comics': 'comic book',
        'religious': 'religion',
        'selfhelp': 'self help',
        'self-help': 'self help',
        'young adult': 'young adult',
        'young adult' : 'young adult contemporary',
        'young adult': 'young adult fantasy',
        'young adult': 'young adult historical fiction',
        'young adult': 'young adult romance'
    }

    for token in doc_user:  # Process each word in the user's input
        token_lower = token.lemma_.lower()  # Get the lemma (base form) of the token in lowercase
        token_lower_text = token.text.lower()  # Get the original form of the token in lowercase
        
        if token_lower_text in conversion_dict:  # If the original word matches a key in the conversion dictionary, use the converted genre
            user_genres.append(conversion_dict[token_lower_text])
        
        elif token.dep_ != 'ROOT':  # If the word is not the main verb (ROOT), try to combine it with the next word to form a compound word
            try:
                compound_word = token_lower + ' ' + token.nbor(1).lemma_.lower()
            except:
                compound_word = None
            
            if compound_word in vocabulary:  # If the compound word exists in the vocabulary, add it to the list
                user_genres.append(compound_word)
            elif token_lower in vocabulary:  # If the single word exists in the vocabulary, add it to the list
                user_genres.append(token_lower)
        elif token_lower in vocabulary:  # If the single word is in the vocabulary, add it to the list
            user_genres.append(token_lower)
    
    user_genres = list(set(user_genres))  # Remove duplicates from the list of user genres
    user_genres_string = ','.join(user_genres)  # Combine the genres into a single string, separated by commas
    
    user_input_vector = vectorizer.transform([user_genres_string])  # Convert the user genres string into a vector using the pre-trained vectorizer
    user_rating_count = recommendation_type_value  # Set the rating count weight (default is 1)
    user_vector = sp.hstack([user_input_vector, [[user_rating_count]]])  # Combine the genre vector with the rating count to form the user vector
    
    distances, indices = knn.kneighbors(user_vector, n_neighbors=int(neighbors))  # Find the nearest neighbors (recommended books) based on the user's vector
    
    recommended_books = df.iloc[indices[0]]  # Select the recommended books from the DataFrame based on the indices
    recommended_books = recommended_books.sort_values(by='Rating', ascending=False)  # Sort the recommended books by rating count in descending order
    
    return recommended_books[['Title', 'Rating', 'Rating Count', 'Genre']]  # Return the recommended books with their title, rating, rating count, and genre



# Set title of the app
st.title('📚 Book Recommender System')

# Add instructions or a description
st.subheader('*Find book recommendations based on your favorite genres.*')

# Sidebar for controls
with st.sidebar:
    st.header('🛠️ Filters')

    user_input_ = st.text_input(
        'Enter Genres:',
        help="You can type in multiple genres.",
        placeholder="Example: fantasy, science fiction..."  # Custom placeholder text
    )

    neighbors = st.slider('Number of recommendations:', 1, 20, 5)  # Slider for number of recommendations
    
    recommendation_type = st.radio(
    "Choose recommendation type:",
    options=["Most Popular", "Most Relevant"]
)
    if recommendation_type == "Most Popular":
        recommendation_type_value = 1
    elif recommendation_type == "Most Relevant":
        recommendation_type_value = 0.5
        
    st.write('')
    button = st.button('Find Books')

#user_input_= ', '.join(user_input_)


st.write('')
st.write('')

# If button clicked, display results
if button:
    st.success("Your Book Recommendations:")
    with st.spinner('Finding the best books for you...'):
        results = recommender(user_input_, neighbors, recommendation_type_value)  # Call recommender function
    
    # Use columns to show results in a better format
    if not results.empty:
        for index, row in results.iterrows():
            st.markdown(f">### **{row['Title']}**")
            st.write(f"⭐ Rating: {row['Rating']} | 📊 Rating Count: {row['Rating Count']}")
        
            # Split the genre string by delimiters like semicolon and comma
            genres = re.split(r'[;,]', row['Genre'])
        
            # Display each genre on a new line
            st.caption("**Genres:** " + ', '.join([genre.strip() for genre in genres]))
        
            st.write("---")  # Add separator between recommendations

    else:
        st.write("No recommendations found.")
else:
    st.error('Please enter some genres in the sidebar.')

# Footer
st.markdown("*🔎 Powered by a KNN machine learning algorithm.*")



