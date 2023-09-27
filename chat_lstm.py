import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle
import random

import colorama
colorama.init()
from colorama import Fore, Style

# Load intents data
with open("intents.json") as file:
    data = json.load(file)

# Load trained LSTM model
model = keras.models.load_model('chat_model_lstm.h5')

# Load tokenizer and label encoder objects
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 20

def get_response(user_input):
    # Tokenize user input
    input_sequence = tokenizer.texts_to_sequences([user_input])
    padded_input_sequence = keras.preprocessing.sequence.pad_sequences(input_sequence, truncating='post', maxlen=max_len)

    # Predict intent
    result = model.predict(padded_input_sequence)
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

    # Get a random response from the matched intent
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def chat():
    print(Fore.YELLOW + "Start messaging with the bot (type 'quit' to stop)!" + Style.RESET_ALL)
    
    while True:
        user_input = input(Fore.LIGHTBLUE_EX + "You: " + Style.RESET_ALL)
        
        if user_input.lower() == "quit":
            print(Fore.YELLOW + "Goodbye! ChatBot: Bye! Have a nice day!" + Style.RESET_ALL)
            break

        response = get_response(user_input)
        print(Fore.GREEN + "ChatBot: " + Style.RESET_ALL + response)

if __name__ == "__main__":
    chat()
