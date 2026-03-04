import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("main_train.csv")  # change path if needed

# ===============================
# 2. Basic Cleaning
# ===============================
# Remove missing values (if any)
df = df.dropna()

# ===============================
# 3. Separate Features and Target
# ===============================
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# ===============================
# 4. Encode Categorical Column (Location)
# ===============================
X = pd.get_dummies(X, columns=["Location"], drop_first=True)

# ===============================
# 5. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 6. Save Final Processed Dataset
# ===============================
processed_df = pd.concat([X, y], axis=1)
processed_df.to_csv("housing_final_processed.csv", index=False)

print("Feature extraction completed successfully!")
print("Final feature count:", X.shape[1])
print("Training shape:", X_train.shape)
