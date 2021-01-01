## Load the dependancies
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pickle

from Data_Preprocessing import clean_lyrics


# Generate text from the trained model
def generate_lyrics(word):
	
	# clean the input
	word = clean_lyrics(word)

	# Tokenize the input
	inputs = np.zeros((1, 1))
	inputs[0, 0] = word2idx[word]
	
	# Predict the first 100 words
	count = 1
	while count <= 10:
		pred = model.predict(inputs)
		word = np.argmax(pred)
		if word >= vocab_size:
			word = vocab_size - 1
			
		inputs[0, 0] = word
		print(idx2word[word], end=" ")
		count += 1



# load the model
model = keras.models.load_model('./Drake_Lyrics_Generator.h5')

# read the tokenizer outputted from the training script
tokenizer = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)

# load the tokenizer data
word2idx = tokenizer.word_index
idx2word = {value: key for key, value in word2idx.items()}

word2idx["<pad>"] = 0
idx2word[0] = "<pad>"

maxlen = 1024
embedding_dim = 128
vocab_size = len(tokenizer.word_index) + 1

print('\nType exit() to exit this application\n')

while True:

	text_input = input("Type a Word to Begin a Verse: ")

	if text_input == 'exit()':
		break
	else:
		print('Drake Would Continue to Say:\n\n\n')
		generate_lyrics(text_input)
		print('\n')
