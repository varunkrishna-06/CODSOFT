# Customer Churn Detection Project

## Overview

This project aims to develop and evaluate a machine learning model to predict customer churn using a dataset of customer information. The target variable indicates whether a customer has exited (churned) or not. Logistic Regression is used due to its suitability for binary classification tasks.

## Steps Involved

### 1. **Loading the Dataset**

The dataset is loaded from a CSV file into a pandas DataFrame, setting up the data for preprocessing and analysis.

### 2. **Data Preprocessing**

- **Label Encoding**: Converts categorical variables (`Gender` and `Geography`) to numerical values using `LabelEncoder`.
- **Feature and Target Separation**: Removes irrelevant columns and separates features from the target variable (`Exited`).

### 3. **Feature Scaling**

- **Standardization**: Scales features using `StandardScaler` to ensure that all features contribute equally to the model.

### 4. **Splitting the Dataset**

- **Train-Test Split**: Divides the data into training and test sets (80% training, 20% test) to evaluate the model’s performance on unseen data.

### 5. **Training the Logistic Regression Model**

- **Model Training**: Trains a Logistic Regression model on the training data to predict customer churn.

### 6. **Model Evaluation**

- **Accuracy**: Measures the proportion of correct predictions.
- **Confusion Matrix**: Displays the number of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Provides precision, recall, and F1-score metrics.
- **ROC-AUC Score**: Evaluates the model’s ability to distinguish between churned and non-churned customers.

### 7. **Hyperparameter Tuning (Optional)**

- **Grid Search**: Finds the best hyperparameters for the model using `GridSearchCV`.
- **Retrain with Best Parameters**: Retrains the model with optimized parameters to potentially improve performance.

### 8. **Visualization**

- **Confusion Matrix**: A heatmap visualizes the model's performance.
- **ROC Curve**: Plots the true positive rate against the false positive rate.
- **Feature Importance Plot**: Uses the absolute values of coefficients to show feature importance.
- **Learning Curves**: Displays training and validation scores as a function of the training size to understand model behavior.
- **Precision-Recall Curve**: Illustrates the trade-off between precision and recall.
- **Histogram of Predicted Probabilities**: Shows the distribution of predicted probabilities.

## Conclusion

The Logistic Regression model effectively predicts customer churn with high accuracy and ROC-AUC scores. Hyperparameter tuning improved the model's performance, and visualizations provided insights into its effectiveness and feature importance. This model is useful for identifying at-risk customers and can be further improved by exploring different algorithms, enhancing feature engineering, and applying it in real-world scenarios for continuous optimization.
