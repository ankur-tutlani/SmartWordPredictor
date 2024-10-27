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
from keras.models import load_model


# In[3]:


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words


# In[4]:


file_path = 'input_dictionary.txt'
words = load_data(file_path)


# In[6]:


from collections import defaultdict, Counter

def split_by_length(words):
    length_dict = defaultdict(list)
    for word in words:
        length_dict[len(word)].append(word)
    return length_dict

def count_forward_backward(words):
    forward_counts = defaultdict(lambda: defaultdict(int))
    backward_counts = defaultdict(lambda: defaultdict(int))
    
    for word in words:
        for i in range(len(word)):
            for j in range(i + 1, len(word)):
                forward_counts[word[i]][word[j]] += 1
            for j in range(i - 1, -1, -1):
                backward_counts[word[i]][word[j]] += 1
    
    return forward_counts, backward_counts


def normalize_counts(counts):
    normalized_counts = {}
    for char, next_chars in counts.items():
        total = sum(next_chars.values())
        normalized_counts[char] = {k: v / total for k, v in next_chars.items()}
    return normalized_counts

def process_words(words):
    length_dict = split_by_length(words)
    results = {}
    
    for length, word_list in length_dict.items():
        forward_counts, backward_counts = count_forward_backward(word_list)
        normalized_forward = normalize_counts(forward_counts)
        normalized_backward = normalize_counts(backward_counts)
        results[length] = {
            'forward': normalized_forward,
            'backward': normalized_backward
        }
    
    return results

results = process_words(words)


# In[9]:


def recommend_letter(masked_word, results):
    length = len(masked_word)
    if length not in results:
        return None
    
    forward_counts = results[length]['forward']
    backward_counts = results[length]['backward']
    
    recommendations = []
    
    for i, char in enumerate(masked_word):
        if char == '0':
            forward_recommendation = None
            backward_recommendation = None
            
            if i > 0 and masked_word[i - 1] != '0':
                prev_char = masked_word[i - 1]
                if prev_char in forward_counts:
                    forward_recommendation = max(forward_counts[prev_char], key=forward_counts[prev_char].get)
            
            if i < length - 1 and masked_word[i + 1] != '0':
                next_char = masked_word[i + 1]
                if next_char in backward_counts:
                    backward_recommendation = max(backward_counts[next_char], key=backward_counts[next_char].get)
            
            if forward_recommendation and backward_recommendation:
                forward_value = forward_counts[prev_char][forward_recommendation]
                backward_value = backward_counts[next_char][backward_recommendation]
                best_recommendation = forward_recommendation if forward_value > backward_value else backward_recommendation
            elif forward_recommendation:
                best_recommendation = forward_recommendation
            elif backward_recommendation:
                best_recommendation = backward_recommendation
            else:
                best_recommendation = None
            
            if best_recommendation:
                recommendations.append((best_recommendation, forward_counts[prev_char][best_recommendation] if forward_recommendation else 0, backward_counts[next_char][best_recommendation] if backward_recommendation else 0))
    
    if recommendations:
        # Find the recommendation with the highest normalized count
        best_recommendation = max(recommendations, key=lambda x: max(x[1], x[2]))[0]
        return best_recommendation
    
    return None


# In[10]:


def count_letter_frequencies(words):
    letter_counts = Counter()
    for word in words:
        letter_counts.update(word)
    return letter_counts


# In[11]:


def create_letter_counts_for_all_lengths(words):
    length_dict = split_by_length(words)
    letter_counts_by_length = {}
    
    for length, word_list in length_dict.items():
        letter_counts_by_length[length] = count_letter_frequencies(word_list)
    
    return letter_counts_by_length


# In[12]:


### in the ratio of their frequencies, and not just the maximum always  #####
def recommend_letter_for_fully_masked(masked_word, letter_counts_by_length):
    length = len(masked_word)
    if length not in letter_counts_by_length:
        return None
    
    letter_counts = letter_counts_by_length[length]
    
    if letter_counts:
        letters = list(letter_counts.keys())
        probabilities = list(letter_counts.values())
        recommended_letter = random.choices(letters, probabilities)[0]
        return recommended_letter
    
    return None


# In[13]:


letter_counts_by_length = create_letter_counts_for_all_lengths(words)


# In[15]:


def is_fully_masked(word, mask_char='0'):
    return all(char == mask_char for char in word)


# In[16]:


def count_letter_frequencies_alldata(words):
    all_letters = ''.join(words)
    return Counter(all_letters)


# In[17]:


##### based on probabilities and not the maximum always ####
def most_frequent_letter(letter_counts_alldata):
    if letter_counts_alldata:
        letters = list(letter_counts_alldata.keys())
        probabilities = list(letter_counts_alldata.values())
        return random.choices(letters, probabilities)[0]
    return None


# In[18]:


letter_counts_alldata = count_letter_frequencies_alldata(words)


# In[19]:


def predict_word(word_with_missing, guessed_letters,results):
    
    if is_fully_masked(word_with_missing, mask_char='0'):
        predicted_char = recommend_letter_for_fully_masked(word_with_missing, letter_counts_by_length)
    else:
        predicted_char = recommend_letter(word_with_missing,results)
    
    if predicted_char not in guessed_letters:
        return predicted_char
    
    most_frequent_letter_recommend =  most_frequent_letter(letter_counts_alldata)
    
    if most_frequent_letter_recommend not in guessed_letters:
        return most_frequent_letter_recommend
    
    
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




