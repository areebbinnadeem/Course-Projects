import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, adjusted_rand_score
from VectorSpaceModel import VectorSpaceModel

# Load and preprocess the data
def load_data():
    data = {
        "Explainable Artificial Intelligence": [1, 2, 3, 7],
        "Heart Failure": [8, 9, 11],
        "Time Series Forecasting": [12, 13, 14, 15, 16],
        "Transformer Model": [17, 18, 21],
        "Feature Selection": [22, 23, 24, 25, 26]
    }
    docs = []
    labels = []
    for label, doc_ids in data.items():
        for doc_id in doc_ids:
            with open(f"ResearchPapers/{doc_id}.txt", "r", encoding="ISO-8859-1") as file:
                docs.append(file.read())
                labels.append(label)
    return docs, labels


docs, labels = load_data()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.3, random_state=42, stratify=labels)


# Instantiate the Vector Space Model
vsm = VectorSpaceModel()

# Transform the training and test data into TF-IDF vectors
X_train_tfidf = vsm.transform_documents_to_tfidf(X_train)
X_test_tfidf = vsm.transform_documents_to_tfidf(X_test)

# Label encoding for class names
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Text classification using k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_tfidf, y_train_encoded)
y_pred_knn = knn_classifier.predict(X_test_tfidf)

# Evaluate k-NN classifier
accuracy_knn = accuracy_score(y_test_encoded, y_pred_knn)
report_knn = classification_report(y_test_encoded, y_pred_knn)

# Text clustering using k-Means algorithm
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train_tfidf)
y_pred_kmeans = kmeans.predict(X_test_tfidf)

# Evaluate k-Means clustering
silhouette = silhouette_score(X_test_tfidf, y_pred_kmeans)
rand_index = adjusted_rand_score(y_test_encoded, y_pred_kmeans)

# Print evaluation metrics
print("K-NN Classifier Evaluation:")
print(f"Accuracy: {accuracy_knn}")
print("Classification Report:\n", report_knn)

print("\nK-Means Clustering Evaluation:")
print(f"Silhouette Score: {silhouette}")
print(f"Adjusted Rand Index: {rand_index}")
