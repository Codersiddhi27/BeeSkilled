import pandas as pd

# Load a public dataset from GitHub
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv("Titanic-Dataset.csv")

print("Dataset loaded successfully.")
print("\nDataFrame info:")
print(df.info())

print("\nBasic descriptive statistics:")
print(df.describe(include="all"))
