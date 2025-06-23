import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the CSV file (ensure 'europe.csv' is in the working directory)
data = pd.read_csv(r"C:\Users\papav\OneDrive\Υπολογιστής\Applied Economics And Data Analysis\big data managment\3d assingment\europe.csv", index_col=0)


# Convert all columns to numeric values
numeric_data = data.apply(pd.to_numeric, errors='coerce')

# Remove potential NA values
numeric_data = numeric_data.dropna()

# Standardize the data for better clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Compute the distance matrix using Euclidean distance
distance_matrix = sch.distance.pdist(scaled_data, metric='euclidean')

# Perform hierarchical clustering using Ward's method
linkage_matrix = sch.linkage(distance_matrix, method='ward')

# Create and plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(linkage_matrix, labels=numeric_data.index, color_threshold=0.7 * max(linkage_matrix[:, 2]))
plt.title("Δενδρόγραμμα Ιεραρχικής Συσταδοποίησης")
plt.xlabel("Χώρες")
plt.ylabel("Απόσταση")
plt.xticks(rotation=90)
plt.show()
