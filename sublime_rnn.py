#general utilities
import pandas as pd 
import numpy as np
#sklearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#keras imports
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Flatten, Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import np_utils

print('reading dataset')
out = pd.read_pickle('single_seq_lyrics2.pkl', compression='gzip')



print('on encoding y...')
#y -> onehot encoded y
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(out.genre)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)


print('creating X sequences...')
#x -> sequences
#this value is the length of the longest song in our dataset.
len_vals = [len(x) for x in out.lyrics]
max_len = max(len_vals)
max_words = 10000

#create tokenizer
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(out.lyrics)

#create sequences
sequences = tok.texts_to_sequences(out.lyrics)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
sequences_matrix = sequences_matrix.reshape(90928, 8196,1)

print(sequences_matrix[0].shape)
#creating test train split
X_train,X_test,Y_train,Y_test = train_test_split(sequences_matrix,y,test_size=0.2)


#building model
def RNN():
    inputs = Input(name='inputs',shape=X_train[0].shape)
    layer = LSTM(40)(inputs)
    layer = Dense(4, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
#create instance of RNN
print('creating model...')
model = RNN()
print(model.summary())
print('compiling...')
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
print("training begun...")
model.fit(X_train,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])



