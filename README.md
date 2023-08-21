# Custom Decision Tree Classifier

This repository contains a custom implementation of a Decision Tree Classifier in Python.

## Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Example](#example)

## Introduction

The Decision Tree Classifier is implemented using the DecisionNode class and the DecisionTreeClassifier class. It supports various parameters like max_depth for controlling the depth of the tree.

## Usage
Use this to create a machine learning model

## Example

To use the custom Decision Tree Classifier in your project, follow these steps:

1. Import the `DecisionTreeClassifier` class:

   ```python
   from decision_tree_classifier import DecisionTreeClassifier
2. initialize the `DecisionTreeClassifier`:
   ```python
   tree = DecisionTreeClassifier(max_depth=10)
3. fit the tree with your data
   ```python
   tree.fit(features, target)

