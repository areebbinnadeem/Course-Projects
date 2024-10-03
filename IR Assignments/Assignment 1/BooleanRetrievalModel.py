#importing the required libraries
import os
import re
from nltk import word_tokenize
from collections import defaultdict
import json


def preprocess_text(text, stop_words):
    # Remove domain name
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text)
    # Remove numbers between words
    text = re.sub(r'(?<=\w)\d+(?=\s\w)', '', text)
    # Convert to lowercase and remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    # Filter out short words and remove single letters followed by a period
    filtered_words = [word for word in filtered_words if len(word) > 1 and not re.match(r'^[a-zA-Z]\.$', word)]
    # Remove special characters () and ?
    filtered_words = [word for word in filtered_words if not re.match(r'^[()?\']+$', word)]
    return filtered_words

def create_indexes(documents):
    inverted_index = defaultdict(list)
    positional_index = defaultdict(lambda: defaultdict(list))
    doc_ids = []  # Initialize an empty list to store document IDs
    for filename, document in zip(os.listdir(directory), documents):
        doc_id = int(filename.split('.')[0])  # Extract document ID from filename
        if document:  # Check if document is not empty
            doc_ids.append(doc_id)  # Append document ID to doc_ids list
            for position, term in enumerate(document):
                inverted_index[term].append(doc_id)
                positional_index[term][doc_id].append(position)
    return inverted_index, positional_index, doc_ids 

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = {line.strip() for line in file}
    return stop_words

directory = 'ResearchPapers'
stopwords_file = 'Stopword-List.txt'
stop_words = load_stopwords(stopwords_file)

documents = []
for i in range(1, 21):
    filename = str(i) + '.txt'
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='latin-1') as file:
            text = file.read()
            preprocessed_text = preprocess_text(text, stop_words)
            documents.append(preprocessed_text)


inverted_index, positional_index, doc_ids = create_indexes(documents)

with open('InvertedIndex.json', 'w') as ij:
    json.dump(inverted_index,ij)
with open("PositionalIndex.json", 'w') as pj:
    json.dump(positional_index, pj)

ij.close();
pj.close();


with open("InvertedIndex.json", 'r') as ii:
    Inverted_index = json.load(ii)

with open("PositionalIndex.json", 'r') as pi:
    Positional_index = json.load(pi)    

ii.close();
pi.close();

def get_posting_list(word, inverted_index):
    return inverted_index.get(word, [])

def intersection(p1, p2):
    return list(set(p1) & set(p2))

def union(p1, p2):
    return list(set().union(p1, p2))

def NOT(p1, p2):
    return list(set(p1) - set(p2))

def get_pos_posting_list(word, positional_index):
    return positional_index.get(word, [])

def docID(plist):
    return plist[0]

def position(plist):
    return plist[1]

def pos_intersect(p1, p2, k):
    answer = []  # Initialize an empty list to store the resulting documents satisfying proximity condition
    if p1 and p2:  # Check if both postings are not empty
        i = j = 0
        while i < len(p1) and j < len(p2):  # Iterate through both postings
            if docID(p1[i]) == docID(p2[j]):  # If the document IDs are the same
                pp1 = position(p1[i])  # Get the positions of terms in document from posting p1
                pp2 = position(p2[j])  # Get the positions of terms in document from posting p2
                ii = jj = 0
                while ii < len(pp1) and jj < len(pp2):  # Iterate through positions of terms in both documents
                    if abs(pp1[ii] - pp2[jj]) <= k:  
                        answer.append(docID(p1[i]))  
                        break
                    elif pp2[jj] > pp1[ii]:  
                        ii += 1
                    else:  
                        jj += 1
                i += 1  
                j += 1  
            elif docID(p1[i]) < docID(p2[j]):  
                i += 1  
            else:  
                j += 1  
                
    return answer 




def query_handler(query, inverted_index, positional_index):
    query = query.split(" ")
    term = query[0]
    documents = get_posting_list(term, inverted_index)

    # Initialize the operator to None
    op = None

    # Initialize the result set with the documents containing the first term
    result_set = documents

    for index in range(1, len(query)):
        if query[index] == "AND":
            op = '&'
        elif query[index] == "OR":
            op = '||'
        elif query[index] == "NOT":
            op = '!'
        elif "/" in query[index]: 
            result_set = ProximityQueryHandler(query[index], positional_index)
        else:
            term = query[index]
            term_postings = get_posting_list(term, inverted_index)
            if op == '&':
                result_set = intersection(result_set, term_postings)
            elif op == '||':
                result_set = union(result_set, term_postings)
            elif op == '!':
                result_set = NOT(result_set, term_postings)

    return result_set

def ProximityQueryHandler(query, positional_index):
    print("Query:", query) 
    proximity = re.findall(r'\d+', query)
    query = query.split(" ")
    token = []
    term = query[0]
    token.append(term)
    
    for i in range(1, len(query)):
        if(query[i] == "AND"):
            operator = "&"
        elif(query[i] == "NOT"):
            operator = "!"
        elif("/" in query[i]):
            proximity_value = int(proximity[0])  
            k = proximity_value
            p1 = get_pos_posting_list(token[0], positional_index)
            p2 = get_pos_posting_list(token[1], positional_index)
            documents = pos_intersect(p1, p2, k)
            token.remove(token[0])
            proximity.remove(proximity[0])
        else:
            if(operator == '&'):
                term = query[i]
                token.append(term)
            elif(operator == '!'):
                term = query[i]
                token.append(term)
        if(len(token) == 3):
            token.remove(token[0])
    print("Documents:", documents)  
    return documents

query = ""    
while True:
    print("~Enter any number to Exit.~")
    query = input("Enter your query: ")
    if query.isdigit():  # Check if query is a digit
        break
    elif '/' in query:
        results = ProximityQueryHandler(query, Positional_index)
        print("Result-Set:", results) 
    else:
        results = query_handler(query, Inverted_index, Positional_index)  
        print("Result-Set:", results) 