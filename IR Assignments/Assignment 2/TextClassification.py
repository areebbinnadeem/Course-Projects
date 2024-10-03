import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.feature_selection import SelectKBest, chi2
from scipy import stats  # Import stats module from SciPy
from VectorSpaceModel import VectorSpaceModel

# Instantiate the Vector Space Model
vsm = VectorSpaceModel()

# Define the document labels based on the provided class names and document lists
doc_labels = {
    "Explainable Artificial Intelligence": [1, 2, 3, 7],
    "Heart Failure": [8, 9, 11],
    "Time Series Forecasting": [12, 13, 14, 15, 16],
    "Transformer Model": [17, 18, 21],
    "Feature Selection": [22, 23, 24, 25, 26]
}

# Create the feature matrix X and label array Y using Vector Space Model
X = []
Y = []
for label, docs in doc_labels.items():
    for doc in docs:
        X.append(vsm.index[str(doc)]['tf-id-frequencies'])
        Y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Data preprocessing: Scaling and Encoding
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)

# Feature selection
selector = SelectKBest(chi2, k=20)
X_selected = selector.fit_transform(X_scaled, Y_encoded)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y_encoded, test_size=0.3, random_state=42, stratify=Y_encoded, shuffle=True)

# Text classification using k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, Y_train)
knn_predictions = knn_classifier.predict(X_test)

# Text clustering using k-Means algorithm
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_selected)
kmeans_cluster_labels = kmeans.labels_

# Evaluation metrics for text classification
precision = precision_score(Y_test, knn_predictions, average='weighted', zero_division=0)
recall = recall_score(Y_test, knn_predictions, average='weighted', zero_division=0)
f1 = f1_score(Y_test, knn_predictions, average='weighted', zero_division=0)
accuracy = accuracy_score(Y_test, knn_predictions)


# Evaluation metrics for text clustering
silhouette = silhouette_score(X_selected, kmeans_cluster_labels)
rand_index = adjusted_rand_score(Y_encoded, kmeans_cluster_labels)

# Print evaluation metrics
print("Text Classification Metrics:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print("\nText Clustering Metrics:")
print(f"Silhouette Score: {silhouette}")
print(f"Adjusted Rand Index: {rand_index}")