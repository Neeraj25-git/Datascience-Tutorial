import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# The dataset contains patient health parameters and a target variable indicating heart disease presence
df = pd.read_csv("heart_disease_dataset.csv")

# Extract features (independent variables) and target variable (dependent variable)
X = df.drop(columns=['Heart_Disease'])  # Features: Age, BP, Cholesterol, etc.
y = df['Heart_Disease']  # Target: 0 (No Heart Disease), 1 (Heart Disease)

# Balance the dataset by undersampling the majority class (No Heart Disease)
num_class_1 = df['Heart_Disease'].value_counts()[1]  # Count of minority class (Heart Disease)
df_class_0 = df[df['Heart_Disease'] == 0].sample(num_class_1, random_state=42)  # Sample equal number from majority class
df_class_1 = df[df['Heart_Disease'] == 1]  # Keep all samples of minority class
df_balanced = pd.concat([df_class_0, df_class_1]).sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset

# Split balanced data into training and test sets (80% training, 20% testing)
X_balanced = df_balanced.drop(columns=['Heart_Disease'])
y_balanced = df_balanced['Heart_Disease']
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Standardize features to improve model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data using the same scaler

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)  # Fit model to training data

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))  # Display precision, recall, F1-score

# Predict heart disease for new patients with given health parameters
new_patients = np.array([[55, 140, 220, 80, 130],  # Moderate risk patient
                         [70, 160, 280, 90, 160],  # High risk patient
                         [40, 120, 180, 70, 100]]) # Low risk patient

# Standardize the new patient data using the previously fitted scaler
new_patients_scaled = scaler.transform(new_patients)

# Make predictions for new patients
predictions = model.predict(new_patients_scaled)
print("Predictions for new patients:", predictions)  # Output 0 (No Disease) or 1 (Disease)