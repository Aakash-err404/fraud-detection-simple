import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def apply_rules(df, col_mapping):
    """
    Apply rule-based heuristics to flag potential fraud.
    Rules:
    1. High amount + late night (between 00:00 and 05:00)
    2. High amount + Weekend
    """
    rule_score = np.zeros(len(df))
    
    amount_col = col_mapping.get('amount')
    
    if amount_col and amount_col in df.columns:
        # Define 'high amount' as > 90th percentile
        try:
            amt_90th = df[amount_col].quantile(0.90)
            
            # Rule 1: High amount + Late night (Hour < 5)
            mask_r1 = (df[amount_col] > amt_90th) & (df['hour_of_day'] < 5)
            rule_score[mask_r1] += 2
            
            # Rule 2: High amount + Weekend
            mask_r2 = (df[amount_col] > amt_90th) & (df['is_weekend'] == 1)
            rule_score[mask_r2] += 1
            
        except Exception as e:
            print(f"Error applying rules: {e}")
            
    return rule_score

def detect_anomalies(df, col_mapping):
    """
    Use Isolation Forest and Statistical methods (Z-score) for unsupervised anomaly detection.
    """
    anomaly_score = np.zeros(len(df))
    amount_col = col_mapping.get('amount')
    
    # 1. Statistical Z-score on amount
    if amount_col and amount_col in df.columns:
        try:
            mean_amt = df[amount_col].mean()
            std_amt = df[amount_col].std()
            if std_amt > 0:
                z_scores = (df[amount_col] - mean_amt) / std_amt
                # Flag z-score > 3
                anomaly_score[z_scores > 3] += 1
        except:
            pass
            
    # 2. Isolation Forest on available numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target if exists
    target_col = col_mapping.get('target')
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
        
    if len(numeric_cols) > 0:
        try:
            # Drop NaNs for IF training
            df_imputed = df[numeric_cols].fillna(df[numeric_cols].median())
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_imputed)
            
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            preds = iso_forest.fit_predict(scaled_data)
            
            # IF returns -1 for outliers, 1 for inliers
            anomaly_score[preds == -1] += 2
        except Exception as e:
            print(f"Error in Isolation Forest: {e}")
            
    return anomaly_score

def train_ml_model(df, col_mapping):
    """
    Train a Logistic Regression model if a target column exists.
    """
    target_col = col_mapping.get('target')
    if not target_col or target_col not in df.columns:
        return None
        
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(target_col)
        
        if len(numeric_cols) == 0:
            return None
            
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df[target_col]
        
        # Simple Logistic Regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(class_weight='balanced', max_iter=500)
        model.fit(X_scaled, y)
        
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        return preds, probs
    except Exception as e:
        print(f"Error in ML Model: {e}")
        return None

def detect_fraud(df, col_mapping):
    """
    Combine all methods to compute a final risk score and prediction.
    """
    df_out = df.copy()
    
    # 1. Rules
    rule_scores = apply_rules(df, col_mapping)
    
    # 2. Anomalies
    anomaly_scores = detect_anomalies(df, col_mapping)
    
    # Total risk score
    total_risk = rule_scores + anomaly_scores
    df_out['Risk_Score'] = total_risk
    
    # 3. Check for Labels (ML)
    ml_results = train_ml_model(df, col_mapping)
    
    if ml_results is not None:
        preds, probs = ml_results
        df_out['Prediction'] = ['Yes' if p == 1 else 'No' for p in preds]
        df_out['Risk_Score'] = probs * 100 # Override risk with ML probability
    else:
        # Fallback to Threshold on Risk Score
        # If total risk >= 2, we consider it Fraud
        df_out['Prediction'] = ['Yes' if r >= 2 else 'No' for r in total_risk]
        
    return df_out
