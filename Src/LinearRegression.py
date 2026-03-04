import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("housing_final_processed.csv")

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Convert bool to int (important)
X = X.astype(int)

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train Model
# ===============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# ===============================
# Evaluate
# ===============================
y_pred = lr_model.predict(X_test)

print("Linear Regression Metrics")
print("-------------------------")
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))

# ===============================
# Save Model
# ===============================
joblib.dump(lr_model, "linear_regression_model.pkl")

print("Linear Regression model saved successfully!")
