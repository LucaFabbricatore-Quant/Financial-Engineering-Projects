# Comparative Analysis for Credit Risk Scoring in the Hotel Industry

## Overview
This project focuses on the development of a robust credit scoring framework to predict corporate default. The analysis compares traditional statistical methods with machine learning architectures, evaluating their predictive accuracy and stability in a high-dimensional financial environment.

## Methodological Framework
The analysis pipeline, implemented in **R**, consists of the following technical phases:

* **Dimensionality Reduction:** Application of **Principal Component Analysis (PCA)** to address multicollinearity among financial ratios and extract the most significant components for default prediction.
* **Traditional Classifiers:** Implementation of **Linear Discriminant Analysis (LDA)** and **Logistic Regression (GLM)** as statistical benchmarks.
* **Neural Network Optimization:** Development of an iterative process for **Artificial Neural Networks (ANN)** with 1 and 2 hidden layers. The script automates the search for the optimal number of nodes by averaging performance over 100 simulations per configuration to ensure statistical stability.
* **Performance Metrics:** Comparative evaluation using **Confusion Matrices**, **ROC Curves**, **AUC**, and the **Gini Coefficient** to measure the discriminatory power and accuracy of the models.

## Technical Requirements
The project requires the following R libraries:
* `neuralnet` and `caret` for model training and evaluation.
* `FactoMineR` for Principal Component Analysis.
* `ROCR` for performance visualization and metrics.

* `openxlsx` and `readxl` for data management.
