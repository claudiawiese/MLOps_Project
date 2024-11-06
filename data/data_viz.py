import pandas as pd 

df = pd.read_parquet("data/dataset_Cramer.parquet")

print(df.head())
print(df.columns)