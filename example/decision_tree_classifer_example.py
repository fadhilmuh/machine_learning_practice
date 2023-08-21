from numpy import random
import sys
import os

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from models.decision_trees import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Generate dummy data
    num_samples = 100   # Number of samples
    num_features = 10   # Number of features
    num_classes = 2     # Number of classes

    # Generate random features
    X = random.rand(num_samples, num_features)

    # Generate random integer class labels
    y = random.randint(low=0, high=num_classes, size=num_samples)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and train the decision tree
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    # Make predictions
    predictions = tree.predict(X_test)
    print("Accuracy: {:.4f}".format(accuracy_score(predictions, y_test)))