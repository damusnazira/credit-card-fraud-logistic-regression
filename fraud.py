import kagglehub
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Download dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)

# Ver archivos dentro
print(os.listdir(path))

# Construir path al archivo CSV
file_path = os.path.join(path, "creditcard.csv")

# Leer archivo
df = pd.read_csv(file_path)

# Basic overview
print(df.head())

# Shape of dataset
print("Shape:", df.shape)

# Data types and nulls
print(df.info())

# Summary statistics
print(df.describe())
#outcome: “There are no missing values in the dataset. The only variable showing a strong signal in the statistical summary is Amount. The mean is significantly higher than the median, which suggests a right-skewed distribution with extreme values pushing the average upward.”
# checking scalation
print(df.describe())
#define x and y
#I tested removing the Time variable and observed no change in model performance, suggesting that it does not add predictive value in this dataset.
#X = df.drop(columns=["Class", "Time"])
X = df.drop(columns=["Class"])
y = df["Class"]
#Applying a log transformation to Amount reduces skewness and compresses extreme values, which can make the relationship between the feature and the target more linear. This helps logistic regression better capture the underlying pattern and improves model stability.
X["Log_Amount"] = np.log1p(X["Amount"])
X = X.drop(columns=["Amount"])

#train and test with stratification

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X_train.copy()
X_test = X_test.copy()
#Since PCA features are already standardized, I scale the Amount variable to ensure all features are on comparable scales, improving model stability and performance.
scaler = StandardScaler()
X_train["Log_Amount"] = scaler.fit_transform(X_train[["Log_Amount"]])
X_test["Log_Amount"] = scaler.transform(X_test[["Log_Amount"]])

# Tuning regularization strength (C) using cross-validation to optimize recall, since detecting fraud is the priority

param_grid = {
    "C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    param_grid,
    scoring="recall",
    cv=5,
    n_jobs=-1
) # dealing with imbalanced dataset

#Since fraud cases are extremely rare, the class weight assigns a much higher importance to them. In this dataset, a fraud observation is weighted almost 300 times more than a non-fraud, ensuring the model pays attention to the minority class.


#train model
grid.fit(X_train, y_train)
#predictions / using best model
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print("Best C:", grid.best_params_)
#evaluation
#Balancing the training set improves learning, but evaluation should still be done on the original distribution to reflect real-world performance. Accuracy remains misleading because it does not capture the model’s ability to detect rare but critical events
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot it
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Save image
plt.title("Confusion Matrix - Fraud Detection")
plt.savefig("confusion_matrix.png")
plt.show()
from sklearn.metrics import precision_recall_curve, auc

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Compute AUC for PR curve
pr_auc = auc(recall, precision)

# Plot
plt.figure()
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

# Save image
plt.savefig("precision_recall_curve.png")
plt.show()