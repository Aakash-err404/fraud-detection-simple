import pandas as pd
import numpy as np
from datetime import datetime

def infer_columns(df):
    """
    Infer key columns like amount, location, category, etc.,
    based on string matching.
    """
    col_mapping = {
        'amount': None,
        'timestamp': None,
        'location': None,
        'category': None,
        'merchant': None,
        'target': None,
        'payment_method': None,
        'user': None
    }
    
    cols = df.columns
    
    for col in cols:
        c = col.lower()
        if 'amount' in c or 'price' in c or 'value' in c:
            col_mapping['amount'] = col
        elif 'time' in c or 'date' in c or 'ts' in c:
            if not col_mapping['timestamp']:
                col_mapping['timestamp'] = col # just pick the first one or combine later
        elif 'loc' in c or 'city' in c or 'country' in c or 'region' in c:
            col_mapping['location'] = col
        elif 'cat' in c or 'type' in c:
            col_mapping['category'] = col
        elif 'merch' in c or 'store' in c:
            col_mapping['merchant'] = col
        elif 'fraud' in c or 'label' in c or 'target' in c or 'class' in c or 'is_fraud' in c:
            col_mapping['target'] = col
        elif 'pay' in c or 'method' in c:
            col_mapping['payment_method'] = col
        elif 'user' in c or 'cust' in c or 'client' in c or 'account' in c:
            col_mapping['user'] = col
            
    # Additional logic for datetime: if there's both a date and time column, we might just use the time one for hour extraction
    date_cols = [c for c in cols if 'date' in c.lower()]
    time_cols = [c for c in cols if 'time' in c.lower()]
    
    if not col_mapping['timestamp']:
        if time_cols:
            col_mapping['timestamp'] = time_cols[0]
        elif date_cols:
            col_mapping['timestamp'] = date_cols[0]
            
    return col_mapping

def engineer_features(df, col_mapping):
    """
    Extract time-based features and other necessary transformations.
    """
    df_feat = df.copy()
    
    ts_col = col_mapping.get('timestamp')
    if ts_col and ts_col in df_feat.columns:
        try:
            # Try parsing datetime
            df_feat[ts_col] = pd.to_datetime(df_feat[ts_col], errors='coerce')
            
            # Extract features
            df_feat['hour_of_day'] = df_feat[ts_col].dt.hour
            df_feat['day_of_week'] = df_feat[ts_col].dt.dayofweek
            df_feat['is_weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Fill NaNs that failed to parse with defaults
            df_feat['hour_of_day'] = df_feat['hour_of_day'].fillna(12).astype(int)
            df_feat['day_of_week'] = df_feat['day_of_week'].fillna(0).astype(int)
            df_feat['is_weekend'] = df_feat['is_weekend'].fillna(0).astype(int)
        except Exception as e:
            print(f"Could not parse timestamp: {e}")
            df_feat['hour_of_day'] = 12
            df_feat['is_weekend'] = 0
    else:
        # Defaults if no timestamp
        df_feat['hour_of_day'] = 12
        df_feat['is_weekend'] = 0
        
    return df_feat
