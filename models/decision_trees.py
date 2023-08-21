class DecisionNode:
    """
    Represents a node in a decision tree.
    
    Attributes:
        feature (int): Feature index to split on.
        threshold (float): Threshold value for numerical features.
        left (DecisionNode): Left subtree.
        right (DecisionNode): Right subtree.
        value (int): Class label for leaf nodes.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # Feature index to split on
        self.threshold = threshold    # Threshold value for numerical features
        self.left = left              # Left subtree
        self.right = right            # Right subtree
        self.value = value            # Class label (for leaf nodes)

class TreeModel:
    """
    Decision Tree Model implementation.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        tree (DecisionNode): Root node of the decision tree.
    """
    from collections import Counter
    import numpy as np
    from sys import getrecursionlimit
    from warnings import warn

    def __init__(self, max_depth:int=None):
        """
        Initializes the DecisionTreeClassifier.

        Parameters:
            max_depth (int, optional): Maximum depth of the decision tree. If None, the tree will grow until purity.
        """
        if max_depth is not None:
            if max_depth <= self.getrecursionlimit():
                self.max_depth = max_depth
            else:
                self.max_depth = self.getrecursionlimit()
                self.warn(f"Maximum depth exceeds the maximum limit. Setting max_depth to maximum limit")
        else:
            self.max_depth = self.getrecursionlimit()

        self.tree = None
    
    def gini(self, labels):
        """
        Calculates the Gini impurity of a set of labels.

        Parameters:
            labels (array-like): Labels of the data samples.

        Returns:
            float: Gini impurity value.
        """
        total_samples = len(labels)
        class_counts = self.Counter(labels)  
        impurity = 1.0
        
        for count in class_counts.values():
            proportion = count / total_samples
            impurity -= proportion ** 2
        
        return impurity
    
    def split(self, X, y, feature, threshold):
        """
        Splits the data based on a feature and a threshold.

        Parameters:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).
            feature (int): Feature index to split on.
            threshold (float): Threshold value for splitting.

        Returns:
            tuple: left_X, left_y, right_X, right_y representing the split data.
        """
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices
        
        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]
        
        return left_X, left_y, right_X, right_y
    
    def find_best_split(self, X, y):
        """
        Finds the best feature and threshold for splitting the data to minimize Gini impurity.

        Parameters:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).

        Returns:
            tuple: best_feature, best_threshold for the split.
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            unique_values = self.np.unique(X[:, feature])  
            for threshold in unique_values:
                left_y = y[X[:, feature] <= threshold]
                right_y = y[X[:, feature] > threshold]
                
                gini_left = self.gini(left_y)
                gini_right = self.gini(right_y)
                
                weighted_gini = (len(left_y) * gini_left + len(right_y) * gini_right) / len(y)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def fit(self, X, y):
        """
        Fits the decision tree to the training data.

        Parameters:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).
        """
        if not isinstance(X,(self.np.ndarray)):
            try:
                X = self.np.asarray(X).astype(self.np.float64)
            except:
                raise TypeError("Data must be a list or array.")
        else:
            X = X.astype(self.np.float64)
        
        if not isinstance(y, (self.np.ndarray)):
            try:
                y = self.np.asarray(y)
            except:
                raise TypeError("Data must be a list or array.")

        if len(X.shape) != 2 or X.shape[1] is None:
            try:
                X = X.reshape(-1,1)
            except:
                raise ValueError("Data must be 2-Dimensional.")
        
        if len(y.shape) != 1:
            try:
                y = y.squeeze()
            except:
                raise ValueError("Targets must be 1-Dimensional")

        print("Fitting data. Please wait ...", end=" ")
        self.tree = self.build_tree(X, y)
        print("Finished!")

    def predict(self, X):
        """
        Predicts the class labels for a set of samples using the decision tree.

        Parameters:
            X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
            list: Predicted class labels for each input sample.
        """
        if not isinstance(X,(self.np.ndarray)):
            try:
                X = self.np.asarray(X).astype(self.np.float64)
            except:
                raise TypeError("Data must be a list or array.")
        else:
            X = X.astype(self.np.float64)

        if len(X.shape) != 2 or X.shape[1] is None:
            try:
                X = X.reshape(-1,1)
            except:
                raise ValueError("Data must be 2-Dimensional.")

        predictions = [self.predict_single(self.tree, sample) for sample in X]
        return predictions
    
    def predict_single(self, node, sample):
        """
        Recursively predicts the class label for a single sample using the decision tree.

        Parameters:
            node (DecisionNode): Current node in the tree.
            sample (array-like): Feature values of a single sample.

        Returns:
            int: Predicted class label.
        """
        if node.value is not None:
            return node.value
        
        if sample[node.feature] <= node.threshold:
            return self.predict_single(node.left, sample)
        else:
            return self.predict_single(node.right, sample)

class DecisionTreeClassifier(TreeModel):
    def __init__(self, max_depth: int = 3):
        """
        Decision Tree Classifier implementation.

        Attributes:
            max_depth (int): Maximum depth of the tree.
                > default value = 3
                if None then will create node until purity::might causing error

            tree (DecisionNode): Root node of the decision tree.
        """
        super().__init__(max_depth)
    
    def build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.

        Parameters:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).
            depth (int): Current depth of the tree.

        Returns:
            DecisionNode: Root node of the built decision tree.
        """
        try:
            if depth == self.max_depth or len(self.np.unique(y)) == 1:  
                try:
                    return DecisionNode(value=self.Counter(y).most_common(1)[0][0])
                except:
                    try:
                        return DecisionNode(value=self.Counter(y).most_common(1)[0])
                    except:
                        raise ValueError("Cannot create tree. Try setting a lower tree depth.")
            
            best_feature, best_threshold = self.find_best_split(X, y)
            
            if best_feature is None:
                try:
                    return DecisionNode(value=self.Counter(y).most_common(1)[0][0])
                except:
                    try:
                        return DecisionNode(value=self.Counter(y).most_common(1)[0])
                    except:
                        raise ValueError("Cannot create tree. Try setting a lower tree depth.")
            
            left_X, left_y, right_X, right_y = self.split(X, y, best_feature, best_threshold)
            
            left_subtree = self.build_tree(left_X, left_y, depth + 1)
            right_subtree = self.build_tree(right_X, right_y, depth + 1)
            
            return DecisionNode(feature=best_feature, threshold=best_threshold,
                                left=left_subtree, right=right_subtree)
        except RecursionError:
            raise RecursionError(f"Maximum recursion depth reached. Cannot extend more nodes. Try to set a lower tree depth.\n Recommended depth: {self.np.max(10, self.getrecursionlimit() // 1e2)} or lower")
    

class DecisionTreeRegressor(TreeModel):
    def __init__(self, max_depth: int = 3):
        """
        Decision Tree Regressor implementation.

        Attributes:
            max_depth (int): Maximum depth of the tree.
                > default value = 3
                if None then will create node until purity::might causing error
                
            tree (DecisionNode): Root node of the decision tree.
        """
        super().__init__(max_depth)

    def build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree.

        Parameters:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).
            depth (int): Current depth of the tree.

        Returns:
            DecisionNode: Root node of the built decision tree.
        """
        try:
            if len(y) == 1:
                return DecisionNode(value=y[0])
            
            if depth == self.max_depth:
                return DecisionNode(value=self.np.mean(y))
            
            best_feature, best_threshold = self.find_best_split(X, y)
            
            if best_feature is None:
                return DecisionNode(value=self.np.mean(y))
            
            left_X, left_y, right_X, right_y = self.split(X, y, best_feature, best_threshold)
            
            left_subtree = self.build_tree(left_X, left_y, depth + 1)
            right_subtree = self.build_tree(right_X, right_y, depth + 1)
            
            return DecisionNode(feature=best_feature, threshold=best_threshold,
                                left=left_subtree, right=right_subtree)
        except RecursionError:
            raise RecursionError("Maximum depth reached limit. Try to set a lower tree depth.")