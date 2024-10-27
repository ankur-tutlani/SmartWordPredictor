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


# Extract the lengths of all string elements in the list
lengths = [len(item) for item in full_dictionary]

# Find the minimum and maximum lengths
min_length = min(lengths)
max_length = max(lengths)

print("Minimum length:", min_length)
print("Maximum length:", max_length)


# In[5]:


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


# In[6]:


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words


# In[7]:


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


aggregate_word_with_missing=[]
aggregate_targets = []


# In[10]:


random_sublists = create_random_sublists()


# In[12]:


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


for j in range(5):
    random_sublists = create_random_sublists()
    word_with_missing=[]
    targets=[]
    for i in range(10):
        result = randomly_select_and_replace(words,i,ratio[i])
        word_with_missing.extend(list(result.values()))
        targets.extend(list(result.keys()))
    aggregate_word_with_missing.append(word_with_missing)
    aggregate_targets.append(targets)
    print(j)


# In[16]:


def combine_unique_lists(list1, list2):
    combined_list1 = []
    combined_list2 = []
    seen_combinations = set()

    for sublist1, sublist2 in zip(list1, list2):
        for item1, item2 in zip(sublist1, sublist2):
            if (item1, item2) not in seen_combinations:
                combined_list1.append(item1)
                combined_list2.append(item2)
                seen_combinations.add((item1, item2))

    return combined_list1, combined_list2


# In[17]:


word_with_missing, targets = combine_unique_lists(aggregate_word_with_missing, aggregate_targets)


# In[40]:


len(word_with_missing),len(targets)


# In[42]:


len(np.unique(targets)),len(np.unique(word_with_missing))


# In[ ]:


from collections import Counter

def find_duplicates(input_list):
    counter = Counter(input_list)
    return [item for item, count in counter.items() if count > 1]


# In[ ]:


print(find_duplicates(word_with_missing))


# In[21]:


# Extract the lengths of all string elements in the list
lengths = [len(item) for item in full_dictionary]

# Find the minimum and maximum lengths
min_length = min(lengths)
max_length = max(lengths)

print("Minimum length:", min_length)
print("Maximum length:", max_length)


# In[22]:


max_word_length = 29


# In[23]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Masking


# In[26]:


char_to_int={'0': 27,
 'a': 1,
 'b': 2,
 'c': 3,
 'd': 4,
 'e': 5,
 'f': 6,
 'g': 7,
 'h': 8,
 'i': 9,
 'j': 10,
 'k': 11,
 'l': 12,
 'm': 13,
 'n': 14,
 'o': 15,
 'p': 16,
 'q': 17,
 'r': 18,
 's': 19,
 't': 20,
 'u': 21,
 'v': 22,
 'w': 23,
 'x': 24,
 'y': 25,
 'z': 26}


# In[29]:


int_to_char={27: '0',
 1: 'a',
 2: 'b',
 3: 'c',
 4: 'd',
 5: 'e',
 6: 'f',
 7: 'g',
 8: 'h',
 9: 'i',
 10: 'j',
 11: 'k',
 12: 'l',
 13: 'm',
 14: 'n',
 15: 'o',
 16: 'p',
 17: 'q',
 18: 'r',
 19: 's',
 20: 't',
 21: 'u',
 22: 'v',
 23: 'w',
 24: 'x',
 25: 'y',
 26: 'z'}


# In[31]:


def preprocess_data(word_with_missing,targets, char_to_int, max_word_length):
    sequences_list = []
    targets_list = []
    for i in range(len(targets)):
        word = targets[i]
        missing_word=word_with_missing[i]
        sequences_list.append([char_to_int[char] if char in char_to_int else char_to_int['0'] for char in missing_word])
        targets_list.append([char_to_int[char] for char in word])
        
    sequences_list = pad_sequences(sequences_list, maxlen=max_word_length, padding='post')
    targets_list = pad_sequences(targets_list, maxlen=max_word_length, padding='post')
    return sequences_list, targets_list


# In[32]:


X, y = preprocess_data(word_with_missing,targets, char_to_int, max_word_length)


# In[33]:


X.shape


# In[34]:


y.shape


# In[35]:


y = to_categorical(y, num_classes=28) # 26 alpha, 1 masked, 1 padding


# In[36]:


model = Sequential()
model.add(Embedding(input_dim=28, output_dim=128, trainable=True))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))))
model.add(Dropout(0.4))
model.add(TimeDistributed(Dense(28, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[37]:


early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)


# In[38]:


model.fit(X, np.array(y), validation_split=0.2, epochs=5, batch_size=256, callbacks=[early_stopping])


# In[76]:


model


# In[39]:


model.save('lstm_model_v1_updated_new_data_logic_seperate_padding_largerdata.h5')


# In[40]:


model.save('lstm_model_v1_updated_new_data_logic_seperate_padding_largerdata.keras')


# In[41]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# In[55]:


def predict_word(model, word_with_missing, char_to_int, int_to_char, max_word_length, guessed_letters):
    word_encoded = [char_to_int[char] if char in char_to_int else char_to_int['0'] for char in word_with_missing]
    word_padded = pad_sequences([word_encoded], maxlen=max_word_length, padding='post')
    
    prediction = model.predict(word_padded)
    predict_vector = prediction[0]
    
    for i, char in enumerate(word_with_missing):
        if char == '0':
            probabilities = predict_vector[i]
            sorted_indices = np.argsort(-probabilities)
            for idx in sorted_indices:
                predicted_char = int_to_char[idx]
                if predicted_char != '0' and predicted_char not in guessed_letters: ## guessed_letters contains set of chars already predicted
                    return predicted_char
    return None  # Return None if all characters are guessed or no suitable character is found


# In[ ]:


#### Example usage


# In[56]:


original_input_word='github'
masked_word='g _ _ h _ b'
word_with_missing = masked_word[::2].replace("_","0")
guessed_letters = set(['g','h','b']) ### chars already present in the word
predicted_char = predict_word(model, word_with_missing, char_to_int, int_to_char, max_word_length, guessed_letters)
print(predicted_char)


# In[ ]:




