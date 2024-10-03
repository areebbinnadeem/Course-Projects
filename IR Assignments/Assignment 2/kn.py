from sklearn.neighbors import KNeighborsClassifier
from VectorSpaceModel import VectorSpaceModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize

# Function to load data based on class names and document IDs
def load_data():
    # Load the vector space data and labels
    data = []
    labels = []
    
    class_documents = {
        "Explainable Artificial Intelligence": [1, 2, 3, 7],
        "Heart Failure": [8, 9, 11],
        "Time Series Forecasting": [12, 13, 14, 15, 16],
        "Transformer Model": [17, 18, 21],
        "Feature Selection": [22, 23, 24, 25, 26]
    }
    
    # Read each document from the specified class and append to data and labels
    for class_name, doc_ids in class_documents.items():
        for doc_id in doc_ids:
            fileName = f"ResearchPapers/{doc_id}.txt"
            try:
                with open(fileName, 'r', encoding='latin-1') as file:
                    text = file.read()
                data.append(vsm.createQueryVector([text]))
                labels.append(class_name)
            except FileNotFoundError:
                # Continue to the next document if it doesn't exist
                pass
            
    return data, labels


# Function to classify text using KNN
def classify_text(X_train, y_train, X_test):
    # Create K-Nearest Neighbors classifier with 5 neighbors
    classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the classifier using the training data
    classifier.fit(X_train, y_train)

    # Predict the class of each document in the test data
    predictions = classifier.predict(X_test)

    return predictions

# Function to evaluate classification performance
def evaluate_classification(y_test, predictions):
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    accuracy = accuracy_score(y_test, predictions)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

# Load the prepared vector space model
vsm = VectorSpaceModel()

# Load the training and test data
X_train, y_train, X_test, y_test = load_data()

# Perform text classification
predictions = classify_text(vsm, X_train, y_train, X_test)

# Evaluate the performance of the classification model
evaluate_classification(y_test, predictions)
