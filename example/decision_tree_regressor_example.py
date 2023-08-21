import sys
import os

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from models.decision_trees import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

if __name__ == '__main__':
    # Load iris dataset
    diabetes = load_diabetes()

    # Set features
    X = diabetes.data

    # Set target
    y = diabetes.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the decision tree
    tree = DecisionTreeRegressor()
    tree.fit(X_train, y_train)

    # Make predictions
    predictions = tree.predict(X_test)
    print("Root Mean Squared Error: {:.4f}".format(mean_squared_error(predictions, y_test, squared=False)))