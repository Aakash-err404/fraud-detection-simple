import pandas as pd
import numpy as np

def clean_column_names(df):
    """Normalize column names to lowercase and replace spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
    return df

def handle_missing_values(df):
    """Handle missing values by imputing median for numeric and mode for categorical."""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

def process_dataset(file):
    """Main function to process the uploaded dataset."""
    df = pd.read_csv(file)
    df = clean_column_names(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = handle_missing_values(df)
    
    return df
