import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from VectorSpaceModel import VectorSpaceModel

def load_data():
    data = []
    labels = []

    class_documents = {
        "Explainable Artificial Intelligence": [1, 2, 3, 7],
        "Heart Failure": [8, 9, 11],
        "Time Series Forecasting": [12, 13, 14, 15, 16],
        "Transformer Model": [17, 18, 21],
        "Feature Selection": [22, 23, 24, 25, 26]
    }

    for class_name, doc_ids in class_documents.items():
        for doc_id in doc_ids:
            # Load TF-IDF vector for each document
            tf_idf_vector = vsm.index[str(doc_id)]['tf-id-frequencies']
            data.append(tf_idf_vector)
            labels.append(class_name)

    return data, labels

# Load the VectorSpaceModel
vsm = VectorSpaceModel()

# Load data and labels
data, labels = load_data()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Print test labels, predictions, and feature values
print("Test Labels:", y_test)
print("Predictions:", y_pred)
print("Feature Values for Test Data:")
for i in range(len(X_test)):
    print("Test Data", i+1, ":", X_test[i])
