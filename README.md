# Custom Machine Learning Model

This repository contains a custom implementation of a Decision Tree Classifier and Regressor in Python. The code contained in this repository is for self-practice purposes only. **Please refrain from using these models for deployment**. I recommend using [scikit-learn's models](https://scikit-learn.org/stable/documentation.html) for production use.

## Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Example](#example)

## Introduction

The models implemented here may not be optimal. The purpose i made this repository is because i am learning to build machine learning algorithms from scratch. If you like to try some models i have built, you can pull this repository and follow some example on how to use it.

## Usage
Use this to create a machine learning model. Again, i recommend using [scikit-learn's models](https://scikit-learn.org/stable/documentation.html) for production use.

## Example

If you want to use the custom `DecisionTreeClassifier` in your project or anything else, follow these steps:
1. Clone this repository to your workspace (if you have not done it)

   ```python
   !git clone https://github.com/fadhilmuh/machine_learning_practice.git

2. Import the model you want. For example, `DecisionTreeClassifier` class:

   ```python
   from machine_learning_practice.models.decision_trees import DecisionTreeClassifier
   ```
   
   or import the whole library
   ```python
   import machine_learning_practice as MLP

3. initialize the `DecisionTreeClassifier`:
   
   ```python
   tree = DecisionTreeClassifier()
   ```

   or if you use from the main library (following the previous step)
   ```python
   tree = MLP.models.decision_trees.DecisionTreeClassifier()

4. fit the tree with your data
   
   ```python
   tree.fit(features, target)

further example is available in the `example` directory.