#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import requests
import random
import string
import secrets
import time
import re
import collections

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


# In[2]:


def build_dictionary(dictionary_file_location):
    text_file = open(dictionary_file_location,"r")
    full_dictionary = text_file.read().splitlines()
    text_file.close()
    return full_dictionary


# In[3]:


full_dictionary_location = "input_dictionary.txt"
full_dictionary = build_dictionary(full_dictionary_location)        
full_dictionary_common_letter_sorted = collections.Counter("".join(full_dictionary)).most_common()


# In[4]:


import numpy as np
from keras.layers import Conv1D, LSTM, Dense, TimeDistributed, Embedding, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Bidirectional
import random


# In[5]:


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words


# In[6]:


file_path = 'input_dictionary.txt'
words = load_data(file_path)


# In[8]:


import random

def create_random_sublists():
    main_list = list(range(len(words)))
    
    # Shuffle the list to ensure randomness
    random.shuffle(main_list)
    
    # Calculate the size of each sublist
    sublist_size = len(main_list) // 10
    remainder = len(main_list) % 10
    
    sublists = []
    start = 0
    
    for i in range(10):
        end = start + sublist_size + (1 if i < remainder else 0)
        sublists.append(main_list[start:end])
        start = end
    
    return sublists


# In[9]:


random_sublists = create_random_sublists()


# In[11]:


import random

def randomly_select_and_replace(words,random_sublists_index,ratio):
    selected_elements = [words[i] for i in random_sublists[random_sublists_index]]
    
    # Function to replace x% of characters in a string with '0'
    def replace_chars(word):
        num_chars_to_replace = max(1, int(len(word) * ratio))
        indices_to_replace = random.sample(range(len(word)), num_chars_to_replace)
        word_list = list(word)
        for index in indices_to_replace:
            word_list[index] = '0'
        return ''.join(word_list)
        
    result = {element: replace_chars(element) for element in selected_elements}
    
    return result


# In[12]:


word_with_missing=[]
targets=[]


# In[13]:


ratio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]


# In[14]:


for i in range(10):
    result = randomly_select_and_replace(words,i,ratio[i])
    word_with_missing.extend(list(result.values()))
    targets.extend(list(result.keys()))


# In[15]:


len(word_with_missing),len(targets)


# In[16]:


len(np.unique(targets))


# In[19]:


# Extract the lengths of all string elements in the list
lengths = [len(item) for item in full_dictionary]

# Find the minimum and maximum lengths
min_length = min(lengths)
max_length = max(lengths)

print("Minimum length:", min_length)
print("Maximum length:", max_length)


# In[18]:


max_word_length = 29


# In[22]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Masking


# In[23]:


### # char for padding 
input_chars = sorted(set(''.join(word_with_missing) + '#'))
output_chars = sorted(set(''.join(targets) + '#'))


# In[26]:


len(input_chars),len(output_chars)


# In[29]:


input_char_to_index = {char: idx for idx, char in enumerate(input_chars)}
output_char_to_index = {char: idx for idx, char in enumerate(output_chars)}


# In[32]:


#### 0 index for padding in both input and output ##


# In[33]:


def preprocess_data(word_with_missing,targets, input_char_to_index,output_char_to_index, max_word_length):
    sequences_list = []
    targets_list = []
    for i in range(len(targets)):
        word = targets[i]
        missing_word=word_with_missing[i]
        sequences_list.append([input_char_to_index[char] if char in input_char_to_index else input_char_to_index['#'] for char in missing_word])
        targets_list.append([output_char_to_index[word[i]] for i, char in enumerate(missing_word) if char == '0']) # using only masked chars in output
    sequences_list = pad_sequences(sequences_list, maxlen=max_word_length, padding='post',value=input_char_to_index['#'])
    targets_list = pad_sequences(targets_list, maxlen=max_word_length, padding='post',value=output_char_to_index['#'])
    return sequences_list, targets_list


# In[34]:


X, y = preprocess_data(word_with_missing,targets, input_char_to_index,output_char_to_index, max_word_length)


# In[35]:


X.shape,y.shape


# In[ ]:


from tensorflow.keras.saving import register_keras_serializable


# In[113]:


@register_keras_serializable()
def masked_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, output_char_to_index['#']), dtype=tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
@register_keras_serializable()
def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, output_char_to_index['#']), dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.int64)  # Ensure y_true is int64
    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, dtype=tf.int64)  # Ensure y_pred is int64
    y_pred = tf.expand_dims(y_pred, axis=-1)
    
    # Check if any predicted value matches any true value
    correct_predictions = tf.reduce_any(tf.equal(y_true, y_pred), axis=-1)
    correct_predictions = tf.cast(correct_predictions, dtype=tf.float32)
    
    return tf.reduce_sum(correct_predictions * mask) / tf.reduce_sum(mask)


# In[47]:


model = Sequential([
    Embedding(input_dim=len(input_chars), output_dim=128,mask_zero=True,trainable=True),
    Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))),
    Dropout(0.4),
    Dense(len(output_chars), activation='softmax')
])


# In[114]:


### test with Conv1D and exluding mask_zero=True option ###
model = Sequential([
    Embedding(input_dim=len(input_chars), output_dim=128,trainable=True),
    Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
    Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))),
    Dropout(0.4),
    Dense(len(output_chars), activation='softmax')
])


# In[115]:


early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)


# In[116]:


model.compile(optimizer='adam', loss=masked_loss, metrics=[masked_accuracy])


# In[117]:


model.fit(X, y, validation_split=0.2, epochs=20, batch_size=128, callbacks=[early_stopping])


# In[118]:


model.save('lstm_model_v2_with_masked.h5')


# In[119]:


model.save('lstm_model_v2_with_masked.keras')


# In[53]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# In[75]:


# Load the model
model = tf.keras.models.load_model('lstm_model_v2_with_masked.keras', custom_objects={'masked_loss': masked_loss,'masked_accuracy':masked_accuracy})


# In[134]:


def predict_word(model, word_with_missing, output_index_to_char, input_char_to_index, max_word_length, guessed_letters):
    word_encoded = [input_char_to_index[char] if char in input_char_to_index else input_char_to_index['0'] for char in word_with_missing]
    word_padded = pad_sequences([word_encoded], maxlen=max_word_length, padding='post',value=input_char_to_index['#'])
    
    prediction = model.predict(word_padded)
    predict_vector = prediction[0]
    
    for i, char in enumerate(word_with_missing):
        if char == '0':
            probabilities = predict_vector[i]
            sorted_indices = np.argsort(-probabilities)
            for idx in sorted_indices:
                predicted_char = output_index_to_char[idx]
                if predicted_char != '#' and predicted_char not in guessed_letters:
                    return predicted_char
    return None  # Return None if all characters are guessed or no suitable character is found


# In[ ]:


#### Example usage


# In[ ]:


original_input_word='github'
masked_word='g _ _ h _ b'
word_with_missing = masked_word[::2].replace("_","0")
guessed_letters = set(['g','h','b']) ### chars already present in the word
predicted_char = predict_word(model, word_with_missing, char_to_int, int_to_char, max_word_length, guessed_letters)
print(predicted_char)


# In[ ]:




