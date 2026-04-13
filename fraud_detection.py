import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("fraud_clean.csv")   # ⚠️ put your actual file name here

# Show columns
#print("Columns:", data.columns)

# Target column
target = "isFraud"

# Drop unnecessary columns
data = data.drop(["nameOrig", "nameDest"], axis=1)

# Convert categorical column to numeric
data["type"] = data["type"].astype("category").cat.codes

# Features and target
X = data.drop(target, axis=1)
y = data[target]

# Reduce size for faster run
X = X.sample(frac=0.1, random_state=1)
y = y.loc[X.index]

# Model
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

# Train
model.fit(X)

# Predict
y_pred = model.predict(X)

# Convert output (-1 → 1, 1 → 0)
y_pred = [1 if i == -1 else 0 for i in y_pred]

# Output
print("\nClassification Report:\n")
print(classification_report(y, y_pred))