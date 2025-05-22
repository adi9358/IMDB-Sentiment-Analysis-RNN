import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')



## load the imdb dataset word index
word_index  = imdb.get_word_index()
reverse_word_index = {value: key for key , value in word_index.items()}


model = load_model('simple_rnn_model.h5')


### step 2 : helper Function

## function to decode reviews
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])


## function to prerprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review 


### streamlit app

import streamlit as st
st.title("IMDB Movie Sentiment Analysis")
st.write('Enter a movie review to classify it as positive or negative.')

## user_input
user_input = st.text_area("Movie review")

if st.button("classify"):

    preprocess_input = preprocess_text(user_input)

    ## make prediction 

    prediction = model.predict(preprocess_input)

    sentiments = 'Positive' if prediction[0][0] >0.5 else "Negative"

    ## display the result
    st.write(f'Sentiments: {sentiments}')
    st.write(f"Prediction Score: {prediction[0][0]}")

else:
    st.write("Please enter a valid moview review")


