from sklearn.neighbors import KNeighborsClassifier
from VectorSpaceModel import VectorSpaceModel
from sklearn.metrics import accuracy_score

def load_data():
    # Load the vector space data and labels from A2
    data = []
    labels = []
    # Read each document from the specified directory
    for docId in range(1, 27):
        fileName = f"ResearchPapers/{docId}.txt"
        try:
            with open(fileName, 'r', encoding='latin-1') as file:
                text = file.read()
            data.append(vsm.createQueryVector([text]))
            labels.append(docId - 1)  # Assign document ID as the label
        except FileNotFoundError:
            # Continue to the next document if it doesn't exist
            pass
    return data, labels

def classify_text(vectorSpaceModel, X_train, y_train, X_test):
    # Create K-Nearest Neighbors classifier with 5 neighbors
    classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the classifier using the training data
    classifier.fit(X_train, y_train)

    # Predict the class of each document in the test data
    predictions = classifier.predict(X_test)

    return predictions


def evaluate_classification(y_test, predictions):
    precision = precision(y_test, predictions)
    recall = recall(y_test, predictions)
    f1_score = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1_score)
    print("Accuracy:", accuracy)


# Load the prepared vector space model
vsm = VectorSpaceModel()

# Load the training and test data
X_train, y_train, X_test, y_test = load_data()    # Load the prepared data (not included in this snippet)

# Perform text classification
predictions = classify_text(vsm, X_train, y_train, X_test)

# Evaluate the performance of the classification model
evaluate_classification(y_test, predictions)