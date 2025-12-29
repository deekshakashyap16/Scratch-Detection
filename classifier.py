import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load CSVs
train = pd.read_csv("train_ratios.csv")
test  = pd.read_csv("test_ratios.csv")

# Use ratio as feature
X_train = train[["scratch_ratio"]].values
y_train = train["severity"].values

X_test = test[["scratch_ratio"]].values
y_test = test["severity"].values

print("Train class distribution:", np.unique(y_train, return_counts=True))
print("Test class distribution:", np.unique(y_test, return_counts=True))

# Train classifier
model = LogisticRegression(multi_class="multinomial", max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Low", "Medium", "High"]
))
