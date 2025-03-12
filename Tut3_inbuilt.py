import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Define the dataset (each row represents a sample, and columns are features)
data_points = np.array([[1, 2], [2, 4], [4, 6], [6, 8]])  # Feature matrix with 2D points
labels = np.array([0, 0, 1, 1])  # Class labels (two categories: 0 and 1)

# Create an LDA model with 1 component (to reduce dimensionality to 1D)
lda_model = LinearDiscriminantAnalysis(n_components=1)

# Fit the LDA model and transform the data onto the new lower-dimensional space
transformed_data = lda_model.fit_transform(data_points, labels)

# Display the transformed data points in the reduced dimension
print("LDA-transformed data:\n", transformed_data)