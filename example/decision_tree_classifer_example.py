from numpy import random
import sys
import os

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from models.decision_trees import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

if __name__ == '__main__':
    # Load iris dataset
    iris = load_iris()

    # Set features
    X = iris.data

    # Set target
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and train the decision tree
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    # Make predictions
    y_pred = tree.predict(X_test)
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))