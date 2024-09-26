
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Load the dataset
bank_customers = pd.read_csv('Churn.csv')

# Initial data exploration
# Show the first rows of the dataset to understand its structure
print(bank_customers.head())

# General information about the DataFrame
print(bank_customers.info())

# Check for duplicates in the data
print("Number of duplicate rows:", bank_customers.duplicated().sum())

# Check for missing values in the columns
print(bank_customers.isnull().sum())

# Replace NaN in the 'Tenure' column with the median of the column
median_tenure = bank_customers['Tenure'].median()
bank_customers['Tenure'].fillna(median_tenure, inplace=True)

# Check again for missing values after imputation
print(bank_customers.isnull().sum())

# Encoding categorical variables using One-Hot Encoding
bank_customers_encoded = pd.get_dummies(bank_customers, columns=['Geography', 'Gender'], drop_first=True)

# Split the data into independent variables (X) and dependent variable (y)
X = bank_customers_encoded.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = bank_customers_encoded['Exited']

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model without addressing class imbalance
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Prediction on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model with classification report
report = classification_report(y_test, y_pred)
print("Classification report for model without class balancing:")
print(report)

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-score:", f1)

# Improve model quality by applying oversampling techniques (SMOTE)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Retrain the model with balanced data
model.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = model.predict(X_test_scaled)

# Evaluate the balanced model
report_resampled = classification_report(y_test, y_pred_resampled)
print("Classification report for the balanced model:")
print(report_resampled)

# Plot ROC-AUC curve
y_proba = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC-ROC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()
plt.show()