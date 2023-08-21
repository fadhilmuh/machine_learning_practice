import sys
import os

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from models.decision_trees import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_wine

if __name__ == '__main__':
    # Load wine dataset
    wine = load_wine()

    # Set features
    X = wine.data

    # Set target
    y = wine.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the decision tree
    tree = DecisionTreeRegressor()
    tree.fit(X_train, y_train)

    # Make predictions
    y_pred = tree.predict(X_test)
    print("Mean Squared Error: {:.4f}".format(mean_absolute_error(y_test, y_pred)))