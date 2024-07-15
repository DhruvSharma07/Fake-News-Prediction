import streamlit as st
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

# Load your pre-trained model
model = load_model('fakenews.h5')

# Initialize PorterStemmer
ps = PorterStemmer()

# Vocabulary size and sentence length (same as used during training)
voc_size = 5000
sent_length = 25

# Function to preprocess input text
def preprocess_text(test):
    corpus_test = []
    review = re.sub('[^a-zA-Z]', ' ',test)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_test.append(review)
    return corpus_test

# Function to predict if the news is fake or real
def predict_news(news):
    corpus = preprocess_text(news)
    onehot_repr_test = [one_hot(words,voc_size)for words in corpus] 
    embedded_docs_test = pad_sequences(onehot_repr_test,padding='pre',maxlen=sent_length)
    X_test = np.array(embedded_docs_test)
    check = model.predict(X_test)
    predicted_class = (check > 0.5).astype(int)
    return predicted_class

# Streamlit app interface
st.title("Fake News Detection")
st.write("Enter the news title below to check if it is fake or real.")

# Input text box
user_input = st.text_area("News Title")

if st.button("Predict"):
    if user_input:
        result = predict_news(user_input)
        if result[0][0] == 1:
            st.write("The news is **Real**.")
        else:
            st.write("The news is **Fake**.")
    else:
        st.write("Please enter a news title to get a prediction.")
