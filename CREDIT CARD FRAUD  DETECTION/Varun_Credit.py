import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv(r"C:\Users\praji\OneDrive\ドキュメント\Project Varun\Credit Card Fraud Detection\fraudTrain.csv")

# Display column names to identify the correct name for the transaction amount
print("Column Names:\n", df.columns)

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Feature scaling for 'amt'
scaler = StandardScaler()
df['amt'] = scaler.fit_transform(df['amt'].values.reshape(-1, 1))

# Drop irrelevant columns
df = df.drop([
    'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 
    'zip', 'lat', 'long', 'dob', 'unix_time', 'trans_num'
], axis=1)

# Apply label encoding to high-cardinality categorical columns
label_encoder = LabelEncoder()
for col in ['merchant', 'job', 'category']:
    df[col] = label_encoder.fit_transform(df[col])

# Apply one-hot encoding to low-cardinality categorical columns
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Separate features and target
X = df.drop(['is_fraud'], axis=1)  # Features
y = df['is_fraud']  # Target (0: legitimate, 1: fraudulent)

# Visualize the distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(df['amt'], kde=True, bins=30, color='purple')
plt.title('Distribution of Scaled Transaction Amounts')
plt.xlabel('Scaled Transaction Amount')
plt.ylabel('Frequency')
plt.show()

# Visualize class distribution before SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(x=y, color='purple')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Legitimate', 'Fraudulent'])
plt.show()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Check class distribution after resampling
print('Resampled class distribution:', np.bincount(y_train_res))

# Visualize class distribution after SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train_res, color='purple')
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Legitimate', 'Fraudulent'])
plt.show()

# Train a Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_res, y_train_res)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Model evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Visualization

# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraudulent'], yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# 3. Feature Importance from Logistic Regression Coefficients
feature_names = X.columns
coefficients = model.coef_[0]

# Create a DataFrame for plotting
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort by absolute coefficient value
feature_importance = feature_importance.sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Feature Importance based on Logistic Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()
