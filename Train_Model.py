## Load the dependancies
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pickle

from Data_Preprocessing import clean_lyrics



## Utility Functions ##

# Call this function to train with TPUs
def train_with_TPUs():

	# detect and init the TPU
	tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
	tf.config.experimental_connect_to_cluster(tpu)
	tf.tpu.experimental.initialize_tpu_system(tpu)

	# instantiate a distribution strategy
	tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

	# instantiating the model in the strategy scope creates the model on the TPU
	with tpu_strategy.scope():
		
		# Define the model
		model = Sequential()
		model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim, mask_zero = True))
		model.add(GRU(units = 1024, return_sequences = True))
		model.add(Dense(vocab_size))
		
		# Compile the model
		model.compile(optimizer = Adam(), loss = SparseCategoricalCrossentropy(from_logits = True))

	# train model normally
	model.fit(x_train, y_train, epochs = 15, verbose = 1)


def train_without_TPUs():

	# Define the model
	model = Sequential()
	model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim, mask_zero = True))
	model.add(GRU(units = 1024, return_sequences = True))
	model.add(Dense(vocab_size))
	
	# Compile the model
	model.compile(optimizer = Adam(), loss = SparseCategoricalCrossentropy(from_logits = True))

	# train model normally
	model.fit(x_train, y_train, epochs = 15, verbose = 1)



## Data Pre-processing ##

# Read in the data
filepath = './data/drake_data.csv'
data = pd.read_csv(filepath)

# split lyrics per chorus into individual lines of lyrics
split_lyric_lines = data['lyrics'].str.rsplit(pat = '\n')
lyric_per_line = split_lyric_lines.apply(pd.Series).stack().reset_index(drop = True)

# Drop unnecessary tags 
pattern_delete = '^((\[|\().*(\]|\)))'
filter = lyric_per_line.str.contains(pattern_delete)
lyric_per_line = lyric_per_line[~filter].reset_index(drop = True)

# Drop punctuation, set to lower case, and correct any abbreviated expressions 
# into the full expression
lyric_per_line = lyric_per_line.apply(lambda line: clean_lyrics(line))

# Remove any empty cells
lyric_per_line = lyric_per_line[lyric_per_line != ''].reset_index(drop = True)

# Splitting text into list of words
lyrics_words = lyric_per_line.apply(lambda line: line.split())

# Print information about input data
# print(lyrics_words.head)
# print(lyrics_words.shape)



## Setup Training ##

# The features or x of the training data will be the text but one 
#index less than the original lyric line
x_train = [line[:-1] for line in lyrics_words]

# The response or the y of the training data will be the last word
# of each lyric line
y_train = [line[1:] for line in lyrics_words]


# Tokenize all of the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lyrics_words)

# Tokenize all of the training data
x_train = tokenizer.texts_to_sequences(x_train)
y_train = tokenizer.texts_to_sequences(y_train)

# Pad the data
word2idx = tokenizer.word_index
idx2word = {value: key for key, value in word2idx.items()}

word2idx["<pad>"] = 0
idx2word[0] = "<pad>"

maxlen = 1024
embedding_dim = 128
vocab_size = len(tokenizer.word_index) + 1

# save the tokenizer for later prediction
with open('tokenizer.pickle', 'wb') as handle:
	pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)


x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
y_train = pad_sequences(y_train, maxlen=maxlen, padding='post', truncating='post')



## Training the Model

# Comment according to your training infrastructure, if you have TPUs then
# train with them! if not then dont :(

# Train with TPUs
# train_with_TPUs()

# or train without TPUs
train_without_TPUs()



## Save the model ##
model.save("Drake_Lyrics_Generator.h5")



