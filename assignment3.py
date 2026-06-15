import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Dataset load
df = pd.read_csv("Titanic-Dataset.csv")

# Missing values handle kar lo (optional but recommended)
df["Age"] = df["Age"].fillna(df["Age"].median())

# -------------------------
# Label Encoding
# -------------------------
le = LabelEncoder()

# Sex column ko encode karna
df["Sex_LabelEncoded"] = le.fit_transform(df["Sex"])

print("Label Encoded 'Sex' Column:")
print(df[["Sex", "Sex_LabelEncoded"]].head())

# -------------------------
# One-Hot Encoding
# -------------------------
onehot_df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print("\nOne-Hot Encoded Columns:")
print(onehot_df.head())