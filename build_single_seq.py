import numpy as np
import pandas as pd
import string 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import np_utils
#sklearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle
#import seaborn as sns
#import matplotlib.pyplot as plt
print("working....")
df = pd.read_pickle('lyrics.pkl', compression='gzip')

genres = ['Pop', 'Indie', 'Jazz', 'Rock']
df = df[(df.genre == 'Pop') | (df.genre == 'Hip-Hop') | (df.genre == 'Metal') | (df.genre == 'Rock')]
pop = df.loc[df['genre']=='Pop'].head(28408)
hiphop = df.loc[df['genre']=='Hip-Hop'].head(28408)
rock = df.loc[df['genre']=='Rock'].head(28408)
metal = df.loc[df['genre']=='Metal'].head(28408)
#really ratchet
output = pop.append(hiphop)
output = output.append(rock)
output = output.append(metal)
output.dropna(inplace=True)
output.reset_index(drop=True, inplace=True)
df = output

#tells us how many of each genre we have
test = df.groupby('genre').nunique()

df.reset_index(drop=True, inplace=True)
# print(df.head())

#preprocess the df without word embeddings.
def preprocess_df(df, save_pickle = False):
        #split and lower all values 
        print('lowering vals and converting songs to lists...')
        df['lyrics'] = df['lyrics'].apply(lambda x: str(x).lower().replace('\n', " ").split(' '))
        #find max value in lyrics
        print('finding max length of a song...')
        max_len = 0
        for lyric in df['lyrics']:
                max_len = max(len(lyric), max_len)
        max_words = 10000
        print('Tokenizing...')
        tok = Tokenizer(num_words=max_words)
        tok.fit_on_texts(df.lyrics)
        sequences = tok.texts_to_sequences(df.lyrics)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
        #check to make sure length is 8196 
        print(len(list(sequences_matrix[0])))
        #reshape to 3-dimensional because thats how Dynamic RNNs like it.
        #90928 = 28408 * 4 (num of each type of song)
        print('reshaping output')
        sequences_matrix = sequences_matrix.reshape(90928, 8196,1)
        print(sequences_matrix.shape)
        if save_pickle:
                print('writing to file...')
                temp = sequences_matrix.reshape(90928, 8196)
                with open('lyrics_clean.txt', 'w') as outfile:
                        np.savetxt(outfile, temp)
                print('saved.')

def read_txt(file_name='lyrics_clean.txt'):
        input_data = np.loadtxt(file_name)
        input_data = input_data.reshape(90928, 8196,1)
        return input_data 

#preprocess_df(df, save_pickle=True)
# temp = read_txt()
# print(temp.head())




def build_targets(df, save_pickle=False):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(df.genre)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        if save_pickle:
                with open('lyrics_target.txt', 'w') as outfile:
                        np.savetxt(outfile, onehot_encoded)
        return onehot_encoded

build_targets(df, save_pickle=True)


