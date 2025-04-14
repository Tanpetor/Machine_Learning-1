# Classification Techniques: SVM and Logistic Regression

## Overview

This project explores the implementation and evaluation of classification algorithms, focusing on Support Vector Machines (SVM) and Logistic Regression. It also delves into probability calibration, feature transformation, and feature selection methods.

## Key Topics

- **Support Vector Machines and Logistic Regression**: Gain insights into the inner workings of these popular classification methods.
- **Probability Calibration**: Learn how to adjust predicted probabilities to better reflect true likelihoods.
- **Feature Engineering**: Explore methods for transforming variables and selecting the most relevant features for your models.
- **Economic Impact Assessment**: Evaluate the economic implications of the model's predictions.

## Project Structure

### Part 1: SVM, Logistic Regression, and Probability Calibration

- **Synthetic Data Generation**: Create synthetic datasets for model training and evaluation.
- **Random Classifier Benchmarking**: Implement a baseline random classifier and evaluate its performance using AUC-ROC and AUC-PR metrics.
- **SVM with Linear Kernel**: Train a Support Vector Machine with a linear kernel, optimizing the regularization parameter \( C \) using cross-validation.
- **Performance Metrics**: Calculate and visualize ROC and PR curves, and compute AUC-ROC and AUC-PR scores for model evaluation.

### Analysis

- **Threshold Impact**: Analyze how varying decision thresholds affect precision, recall, and overall model performance.
- **Model Comparisons**: Compare the performance of different models and classifiers using a metrics dataframe.
