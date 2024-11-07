import pandas as pd 

df = pd.read_parquet("data/dataset_Cramer.parquet")

print(df.head())
print(df.columns)

df_for_prediction = df.head()

# Save to CSV
df_for_prediction.to_csv("data/sample_data_for_prediction.csv", index=False)

# Save to Parquet
df_for_prediction.to_parquet("data/sample_data_for_prediction.parquet")

