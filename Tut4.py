import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None, n_jobs=-1):
        """
        Initialize the Random Forest.
        
        Parameters:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of each tree
        - max_features: Number of features to consider at each split ('sqrt' for square root of total features)
        - random_state: Seed for reproducibility
        - n_jobs: Number of parallel jobs (-1 uses all processors)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []
        self.feature_indices = []

    def _train_tree(self, X, y, random_state):
        """Trains a single decision tree on a bootstrap sample."""
        np.random.seed(random_state)
        X_sample, y_sample = resample(X, y, random_state=random_state)
        n_features = X.shape[1]
        
        max_features = int(np.sqrt(n_features)) if self.max_features == 'sqrt' else self.max_features
        features = np.random.choice(n_features, max_features, replace=False)
        
        tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=random_state)
        tree.fit(X_sample[:, features], y_sample)
        
        return tree, features

    def fit(self, X, y):
        """
        Train the Random Forest on the given data.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Target vector (n_samples,)
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_tree)(X, y, self.random_state + i if self.random_state else None)
            for i in range(self.n_estimators)
        )

        self.trees, self.feature_indices = zip(*results)

    def predict_proba(self, X):
        """
        Predict class probabilities by averaging predictions from all trees.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        
        Returns:
        - proba: Array of shape (n_samples, n_classes) with class probabilities
        """
        probas = np.mean(
            [tree.predict_proba(X[:, features]) for tree, features in zip(self.trees, self.feature_indices)],
            axis=0
        )
        return probas

    def predict(self, X):
        """
        Predict class labels by taking the majority vote from all trees.
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        
        Returns:
        - pred: Array of shape (n_samples,) with class predictions
        """
        predictions = np.array(
            [tree.predict(X[:, features]) for tree, features in zip(self.trees, self.feature_indices)]
        )
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

# Example usage
if __name__ == "__main__":
    # Load sample dataset
    data = load_iris()
    X, y = data.data, data.target
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest
    rf = RandomForest(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
