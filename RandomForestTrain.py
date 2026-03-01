import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("housing_final_processed.csv")

# ===============================
# Separate Features and Target
# ===============================
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Convert boolean columns to int
X = X.astype(int)

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train Random Forest
# ===============================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Evaluation
# ===============================
y_pred = model.predict(X_test)

print("Random Forest Metrics")
print("----------------------")
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))

# ===============================
# Save Model & Columns
# ===============================
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nModel and column structure saved successfully!")