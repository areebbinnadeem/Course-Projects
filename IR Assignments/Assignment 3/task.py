import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

# Reading the CSV file containing TF-IDF values
df = pd.read_csv('tf-idf.csv')

df_transposed = df.T # Taking a transpose of the DataFrame to have documents as rows and features as columns
df_transposed.columns = df_transposed.iloc[0].astype(str)
df_transposed = df_transposed[1:]

# Document labels 
doc_labels = {
    "Explainable Artificial Intelligence": [1, 2, 3, 7],
    "Heart Failure": [8, 9, 11],
    "Time Series Forecasting": [12, 13, 14, 15, 16],
    "Transformer Model": [17, 18, 21],
    "Feature Selection": [22, 23, 24, 25, 26]
}

# Flatten the document labels and encode 
all_doc_ids = [doc_id for doc_ids in doc_labels.values() for doc_id in doc_ids]
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform([label for label, doc_ids in doc_labels.items() for _ in doc_ids])

X_train, X_test, y_train, y_test = train_test_split(df_transposed, Y, test_size=0.1, random_state=42)

# KNN Classifier with hyperparameter tuning
param_grid_knn = {'n_neighbors': [2, 3, 5], 'weights': ['uniform', 'distance']}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, verbose=1)
grid_search_knn.fit(X_train, y_train)

print("Best Parameters for KNN:", grid_search_knn.best_params_)

# Making predictions and evaluate KNN model
y_pred_knn = grid_search_knn.predict(X_train)
accuracy_knn = accuracy_score(y_train, y_pred_knn)
print(f'Training Accuracy for KNN: {accuracy_knn}')
report_knn = classification_report(y_train, y_pred_knn)
print('Training Classification Report for KNN:\n', report_knn)

# KMeans Clustering with hyperparameter tuning
param_grid_kmeans = {'n_clusters': [2, 3, 5, 7 ,10]}
kmeans = KMeans(random_state=42)
grid_search_kmeans = GridSearchCV(kmeans, param_grid_kmeans, cv=5, verbose=1)
grid_search_kmeans.fit(X_train)

print("Best Parameters for KMeans:", grid_search_kmeans.best_params_)

# Making predictions and evaluate KMeans model
y_pred_kmeans = grid_search_kmeans.predict(X_train)

clusters = np.zeros_like(y_pred_kmeans)
for i in range(len(np.unique(y_pred_kmeans))):
    mask = (y_pred_kmeans == i)
    clusters[mask] = np.bincount(y_train[mask]).argmax()

purity = np.sum(clusters == y_train) / y_train.shape[0]
print(f'Purity for KMeans: {purity}')

silhouette = silhouette_score(X_train, y_pred_kmeans)
print(f'Silhouette Score for KMeans: {silhouette}')

rand_index = adjusted_rand_score(y_train, y_pred_kmeans)
print(f'Adjusted Rand Index for KMeans: {rand_index}')

import tkinter as tk
from tkinter import scrolledtext

class ResultsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Results")

        self.results_text = scrolledtext.ScrolledText(self.root, width=60, height=20, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.display_results()

    def display_results(self):
        self.results_text.insert(tk.END, "KNN Results:\n")
        self.results_text.insert(tk.END, f"Best Parameters for KNN: {grid_search_knn.best_params_}\n")
        self.results_text.insert(tk.END, f"Training Accuracy for KNN: {accuracy_knn}\n")
        self.results_text.insert(tk.END, f"Training Classification Report for KNN:\n{report_knn}\n\n")

        self.results_text.insert(tk.END, "KMeans Results:\n")
        self.results_text.insert(tk.END, f"Best Parameters for KMeans: {grid_search_kmeans.best_params_}\n")
        self.results_text.insert(tk.END, f"Purity for KMeans: {purity}\n")
        self.results_text.insert(tk.END, f"Silhouette Score for KMeans: {silhouette}\n")
        self.results_text.insert(tk.END, f"Adjusted Rand Index for KMeans: {rand_index}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResultsGUI(root)
    root.mainloop()

