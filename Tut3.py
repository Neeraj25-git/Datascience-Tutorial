import numpy as np

# Define the feature matrix (each row represents a sample, columns are features)
data_samples = np.array([[1, 2], [2, 4], [4, 6], [6, 8]])  # 4 samples with 2 features
class_labels = np.array([0, 0, 1, 1])  # Class labels (two categories: 0 and 1)

# Separate the samples based on their class labels
class_A = data_samples[class_labels == 0]  # Extract samples belonging to class 0
class_B = data_samples[class_labels == 1]  # Extract samples belonging to class 1

# Compute the mean vectors for each class (column-wise mean)
mean_A = np.mean(class_A, axis=0)  # Mean vector of class 0
mean_B = np.mean(class_B, axis=0)  # Mean vector of class 1

# Compute the overall mean of all samples (not used in this implementation, but useful for reference)
overall_mean = np.mean(data_samples, axis=0)  # Overall mean vector

# Compute the within-class scatter matrix (S_W)
# S_W captures the spread of data within each class
S_W = np.dot((class_A - mean_A).T, (class_A - mean_A)) + np.dot((class_B - mean_B).T, (class_B - mean_B))

# Compute the between-class scatter matrix (S_B)
# S_B captures the difference between the class means
mean_difference = (mean_A - mean_B).reshape(-1, 1)  # Convert mean difference into a column vector
S_B = np.dot(mean_difference, mean_difference.T)  # Outer product to compute S_B

# Solve the eigenvalue problem for inv(S_W) * S_B
# Finding the eigenvectors that define the best projection direction
eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Select the eigenvector corresponding to the largest eigenvalue
lda_projection_vector = eigen_vectors[:, np.argmax(eigen_values)]  # Optimal projection direction

# Project the original data onto the LDA projection vector
transformed_data_manual = np.dot(data_samples, lda_projection_vector)

# Display the transformed data points
print("LDA-transformed data using manual computation:\n", transformed_data_manual)