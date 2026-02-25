# ==========================================
# House Price Prediction - Random Forest
# Train + Evaluate + Save Pickle
# ==========================================

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------
# 1. Load Data
# ------------------------------

train_data = pd.read_csv("House price datasets/train.csv")
test_data = pd.read_csv("House price datasets/test.csv")

TARGET = "SalePrice"

# ------------------------------
# 2. Preprocessing
# ------------------------------

X = train_data.drop(columns=[TARGET])
y = np.log1p(train_data[TARGET])   # Log transform target

# Combine for consistent preprocessing
combined = pd.concat([X, test_data], axis=0)

# Fill missing values
num_cols = combined.select_dtypes(include=np.number).columns
combined[num_cols] = combined[num_cols].fillna(combined[num_cols].median())

cat_cols = combined.select_dtypes(exclude=np.number).columns
combined[cat_cols] = combined[cat_cols].fillna(combined[cat_cols].mode().iloc[0])

# One-hot encoding
combined = pd.get_dummies(combined)

# Split back
X = combined[:len(train_data)]
X_test = combined[len(train_data):]

# ------------------------------
# 3. Train / Validation Split
# ------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 4. Train Random Forest
# ------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------------
# 5. Evaluation
# ------------------------------

y_pred = np.expm1(model.predict(X_val))
y_true = np.expm1(y_val)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("----- Random Forest Model Evaluation -----")
print(f"MAE  : {mae}")
print(f"MSE  : {mse}")
print(f"RMSE : {rmse}")
print(f"R2   : {r2}")

# ------------------------------
# 6. Save Model & Columns
# ------------------------------

pickle.dump(model, open("rf_model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("\nRandom Forest model saved successfully!")
print("Files created:")
print(" - rf_model.pkl")
print(" - columns.pkl")