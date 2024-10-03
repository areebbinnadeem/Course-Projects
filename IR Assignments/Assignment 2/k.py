import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from VectorSpaceModel import VectorSpaceModel

# Load the VectorSpaceModel
vsm = VectorSpaceModel()

# Load the data and labels using the VectorSpaceModel
docs = []
labels = []

class_names = {
    "Explainable Artificial Intelligence": [1, 2, 3, 7],
    "Heart Failure": [8, 9, 11],
    "Time Series Forecasting": [12, 13, 14, 15, 16],
    "Transformer Model": [17, 18, 21],
    "Feature Selection": [22, 23, 24, 25, 26]
}

for class_name, doc_ids in class_names.items():
    for doc_id in doc_ids:
        docs.append(vsm.index[str(doc_id)]['tf-id-frequencies'])
        labels.append(class_name)

# Convert lists to numpy arrays
X = np.array(docs)
Y = np.array(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the KNN classifier with hyperparameter tuning
param_grid = {'n_neighbors': [3, 5, 7, 9]}  # Example grid of hyperparameters
knn_classifier = KNeighborsClassifier()
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = grid_search.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
