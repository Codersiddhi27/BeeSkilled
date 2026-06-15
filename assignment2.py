import pandas as pd

# Dataset load
df = pd.read_csv("Titanic-Dataset.csv")

# Missing values check
print("Missing values before imputation:")
print(df.isnull().sum())

# Numerical columns select karo
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Mean Imputation
df_mean = df.copy()
for col in numeric_cols:
    df_mean[col] = df_mean[col].fillna(df_mean[col].mean())

# Median Imputation
df_median = df.copy()
for col in numeric_cols:
    df_median[col] = df_median[col].fillna(df_median[col].median())

print("\nMissing values after Mean Imputation:")
print(df_mean[numeric_cols].isnull().sum())

print("\nMissing values after Median Imputation:")
print(df_median[numeric_cols].isnull().sum())