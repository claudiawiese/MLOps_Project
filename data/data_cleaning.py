import pandas as pd

def clean_accident_data(file_path: str, output_path: str) -> pd.DataFrame:
    # Load the raw data
    df = pd.read_csv(file_path)

    # Convert the 'an' column to string and standardize the year format
    df['an'] = df['an'].astype(str)
    df['an'] = df['an'].apply(lambda x: '200' + x if len(x) == 1 else x)
    df['an'] = df['an'].apply(lambda x: '20' + x if len(x) == 2 else x)

    # Filter data for years starting from 2019
    df = df[df['an'] >= "2019"].drop_duplicates()

    # Remove administrative features
    features_to_remove_admin = ["adr", "pr", "pr1", "id_vehicule_x", "id_usager", "id_vehicule_y", "Num_Acc"]
    df.drop(columns=features_to_remove_admin, inplace=True)

    # Remove features with missing data after 2019
    features_to_remove_no_data_after_2019 = ["lartpc", "larrout", "gps", "env1", "secu", "occutc"]
    df.drop(columns=features_to_remove_no_data_after_2019, inplace=True)

    # Drop date-related columns
    df.drop(columns=["date", "hrmn"], inplace=True)

    # Handle missing values (-1) and zeros
    df.replace(-1, pd.NA, inplace=True)

    # Drop columns with more than 2% unique values
    len_total = df.shape[0]
    for col in df.columns:
        if (df[col].nunique() / len_total) >= 0.02:
            df.drop(columns=[col], inplace=True)

    # Export cleaned DataFrame to a parquet file
    df.to_parquet(output_path, index=False)

    return df

# Example usage
cleaned_data = clean_accident_data(
    file_path="data/raw_data_merged_accident.csv",
    output_path="data/cleaned_accident_data.parquet"
)
