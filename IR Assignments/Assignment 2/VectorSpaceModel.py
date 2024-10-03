import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log, sqrt

class VectorSpaceModel:
    def __init__(self):
        self.totalNumberOfDocuments = 20
        self.stopwords = self.load_stopwords("Stopword-list.txt")  
        self.stemmer = PorterStemmer()  
        # Create index for the vector space model
        self.index = self.createIndex()  
    
    def load_stopwords(self, filename):
        with open(filename, "r") as file:
            stopwords = file.read().splitlines()
        return set(stopwords)

    def createIndex(self):
        index = {}  
        for docId in range(1, self.totalNumberOfDocuments + 1):
            fileName = f"ResearchPapers/{docId}.txt"  
            if os.path.exists(fileName):
                fileTokens = self.getTokensFromFile(fileName)  
                for word in fileTokens:
                    word = word.lower() 
                    # Checking if word is not a stopword then stem it using Porter Stemmer
                    if word not in self.stopwords:  
                        word = self.stemmer.stem(word)  
                        if word not in index:
                            # Initializing index entry for word
                            index[word] = {'termFrequencies': [0] * self.totalNumberOfDocuments,
                                           'documentFrequency': 0, 'idf': 0, 'tf-id-frequencies': [0] * self.totalNumberOfDocuments}
                        index[word]['termFrequencies'][docId - 1] += 1  # Incrementing term frequency for document
                        index[word]['documentFrequency'] += 1  # Incrementing document frequency for word
        # Calculating IDF and TF-IDF for each word in index
        for word in index:
            index[word]['idf'] = log(self.totalNumberOfDocuments / index[word]['documentFrequency'], 10)
            for i in range(self.totalNumberOfDocuments):
                index[word]['tf-id-frequencies'][i] = index[word]['termFrequencies'][i] * index[word]['idf']
        return index

    def getTokensFromFile(self, fileName):
        with open(fileName, 'r', encoding='latin-1') as file:
            text = file.read()
            return word_tokenize(text)

    def createQueryVector(self, queryTerms):
        queryVector = {}  
        for word in queryTerms:
            word = word.lower()  
            # Checking if word is not a stopword then stem it using Porter Stemmer
            if word not in self.stopwords:  
                word = self.stemmer.stem(word)  
                if word in self.index:
                    if word not in queryVector:
                        # Initializing query vector entry for word
                        queryVector[word] = {'termFrequency': 0, 'tf-id-frequency': 0}
                    queryVector[word]['termFrequency'] += 1  # Incrementing term frequency for query
                    queryVector[word]['tf-id-frequency'] = queryVector[word]['termFrequency'] * self.index[word]['idf']  # Calculating TF-IDF for query
        return queryVector

    def dotProduct(self, queryVector, document):
        # Computing dot product between query vector and document vector
        sumOfProducts = 0
        for word in queryVector:
            sumOfProducts += (queryVector[word]['tf-id-frequency'] * self.index[word]['tf-id-frequencies'][document])
        return sumOfProducts

    def magnitudeProduct(self, queryVector, document):
        # Computing magnitude product for normalization
        queryMagnitude = sum(pow(queryVector[word]['tf-id-frequency'], 2) for word in queryVector)
        documentMagnitude = sum(pow(self.index[word]['tf-id-frequencies'][document], 2) for word in queryVector)
        return sqrt(queryMagnitude * documentMagnitude)

    def cosineSimilarity(self, queryVector):
        # Computing cosine similarity between query vector and each document vector
        similarityValues = []
        for i in range(self.totalNumberOfDocuments):
            try:
                similarity = self.dotProduct(queryVector, i) / self.magnitudeProduct(queryVector, i)
                similarityValues.append((similarity, i + 1))
            except ZeroDivisionError:
                pass
        similarityValues.sort(reverse=True)  # Sorting similarity values in descending order
        return similarityValues

    def filterDocuments(self, documents, alphaValue):
        # Filter documents based on similarity threshold (alpha)
        return [(similarity, document) for similarity, document in documents if similarity >= alphaValue]

    def executeQuery(self, query):
        parsedQuery = word_tokenize(query.lower())
        alphaValue = 0.05
        try:
            alphaValue = float(parsedQuery[-1])
            parsedQuery = parsedQuery[:-1]
        except ValueError:
            pass
        queryVector = self.createQueryVector(parsedQuery)
        similarityValues = self.cosineSimilarity(queryVector)
        documents = self.filterDocuments(similarityValues, alphaValue)
        return documents


def main():
    vsm = VectorSpaceModel()
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = vsm.executeQuery(query)
        if results:
            print("Result-Set:", ','.join(str(doc) for _, doc in results))
        else:
            print("Result-Set: NIL")

if __name__ == "__main__":
    main()