import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# -----------------------------
# 1. Handle Missing Data
# -----------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# -----------------------------
# 2. Encode Sex Column
# -----------------------------
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

# -----------------------------
# 3. Encode Embarked Column
# -----------------------------
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Convert True/False to 1/0 (optional)
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# -----------------------------
# 4. Visualize Age Distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# 5. Save Cleaned Dataset
# -----------------------------
df.to_csv("Titanic_Cleaned.csv", index=False)

print("Dataset cleaned and saved as Titanic_Cleaned.csv")