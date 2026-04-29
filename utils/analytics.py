import matplotlib.pyplot as plt
import pandas as pd

def plot_fraud_distribution(df):
    """Plot Pie chart for Fraud vs Non-Fraud using Matplotlib."""
    counts = df['Prediction'].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#ff9999' if idx == 'Yes' else '#99ff99' for idx in counts.index]
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Fraud vs Non-Fraud Distribution')
    
    return fig

def plot_amount_distribution(df, col_mapping):
    """Plot histogram of transaction amounts, colored by Prediction using Matplotlib."""
    amount_col = col_mapping.get('amount')
    if not amount_col or amount_col not in df.columns:
        return None
        
    # Cap amount at 95th percentile for better visualization
    amt_95 = df[amount_col].quantile(0.95)
    df_plot = df[df[amount_col] <= amt_95]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    fraud_amts = df_plot[df_plot['Prediction'] == 'Yes'][amount_col]
    non_fraud_amts = df_plot[df_plot['Prediction'] == 'No'][amount_col]
    
    ax.hist([non_fraud_amts, fraud_amts], bins=30, stacked=True, 
            color=['#99ff99', '#ff9999'], label=['No', 'Yes'])
            
    ax.set_title('Transaction Amount Distribution (Capped at 95th Percentile)')
    ax.set_xlabel('Amount')
    ax.set_ylabel('Frequency')
    ax.legend(title='Prediction')
    
    return fig

def plot_user_segmentation(df, col_mapping):
    """Segment users into Low, Regular, High spenders based on amount using Matplotlib."""
    user_col = col_mapping.get('user')
    amount_col = col_mapping.get('amount')
    
    if not user_col or not amount_col or user_col not in df.columns or amount_col not in df.columns:
        return None
        
    user_spend = df.groupby(user_col)[amount_col].sum().reset_index()
    
    p33 = user_spend[amount_col].quantile(0.33)
    p66 = user_spend[amount_col].quantile(0.66)
    
    def get_segment(amt):
        if amt <= p33:
            return 'Low Spender'
        elif amt <= p66:
            return 'Regular Spender'
        else:
            return 'High Spender'
            
    user_spend['Segment'] = user_spend[amount_col].apply(get_segment)
    counts = user_spend['Segment'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind='bar', ax=ax, color=['#ffcc99', '#99ccff', '#cc99ff'])
    
    ax.set_title('User Segmentation by Spending Behavior')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Count')
    plt.xticks(rotation=0)
    
    return fig
