import json
import pandas as pd
import numpy as np

with open('InvertedIndex.json') as f: # Loading the inverted index file which contains term frequencies
    term_frequencies = json.load(f)

# Creating a set of all document IDs
doc_ids = set()
for frequencies in term_frequencies.values():
    doc_ids.update(frequencies.keys())

dict = {} # Initializing a new dictionary 

for word, frequencies in term_frequencies.items():
    dict[word] = {} 
    for doc_id in doc_ids: # For each document ID, checking if the word appears in that document
        # If it does, add the frequency to the new dictionary
        if doc_id in frequencies:
            dict[word][doc_id] = frequencies[doc_id]
        else:
            dict[word][doc_id] = 0

# Converting the dictionary into DataFrame
df = pd.DataFrame.from_dict(dict, orient='index')

total_words = df.shape[0] # total no. of unique words

df["term_frequency"] = df.sum(axis=1) # term frequency

df['number_of_document_containing_word'] = 20 - ((df == 0).sum(axis=1))

tf = {}
for col in df.columns:
    if col != "term_frequency":
        tf[col] = (df[col] == 0).sum()
        total = total_words - tf[col]
        tf[col] = total + total_words

for col in df.columns:
    if col != "term_frequency":
        df[col] = (df[col] + 1) / tf[col]

# Calculating idf
idf = (20 + 1) / (df['number_of_document_containing_word'] + 1)
df["idf"] = np.log(idf)

# Calculating tf-idf
tf_idf = pd.DataFrame()
for col in df.columns[:20]:
    tf_idf[col] = df[col] * df["idf"]

tf_idf = tf_idf.reindex(df.index)

tf_idf.to_csv('tf-idf.csv')
