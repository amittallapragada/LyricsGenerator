import numpy as np
import pandas as pd
import string 
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
print("working...")
df = pd.read_pickle('lyrics.pkl', compression='gzip')

genres = ['Pop', 'Indie', 'Jazz', 'Rock']
df = df[(df.genre == 'Pop') | (df.genre == 'Hip-Hop') | (df.genre == 'Metal') | (df.genre == 'Rock')]

#tells us how many of each genre we have
test = df.groupby('genre').nunique()
print(test)



def preprocess_song(lyrics):
    lyrics = lyrics.lower()
    table = lyrics.maketrans({key: None for key in string.punctuation})
    lyrics = lyrics.translate(table)
    lyrics = lyrics.replace('\n', " ")
    lyrics = lyrics.split(' ')
    #for some reason rstrip didnt work in all cases...
    return lyrics


def create_df(df):
    output = defaultdict(list)
    for i in range(len(df)):
        song = preprocess_song(df.loc[i].lyrics)
        genre = df.loc[i].genre
        


#create_df(df)

