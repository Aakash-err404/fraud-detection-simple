import streamlit as st
import pandas as pd
import time

from utils.data_processor import process_dataset
from utils.feature_engine import infer_columns, engineer_features
from utils.fraud_detector import detect_fraud
from utils.analytics import plot_fraud_distribution, plot_amount_distribution, plot_user_segmentation

st.set_page_config(page_title="Fraud Detection System", page_icon="🕵️", layout="wide", initial_sidebar_state="expanded")

st.title("🕵️ Intelligent Fraud Detection System")
st.markdown("""
Upload a transaction dataset (CSV) to identify potential fraudulent activities. 
The system uses a hybrid approach of **Rule-based heuristics**, **Anomaly Detection**, and **Machine Learning** to find suspicious transactions.
""")

st.markdown("💡 **Tip:** Supports any transaction dataset. No fixed schema required.")
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], help="Upload your transaction dataset here.")

if uploaded_file is not None:
    with st.spinner('⏳ Analyzing transactions and identifying patterns...'):
        time.sleep(1) # simulate processing time for UX
        
        # 1. Process Data
        df = process_dataset(uploaded_file)
        
        # 2. Infer Columns
        col_mapping = infer_columns(df)
        
        # 3. Engineer Features
        df_feat = engineer_features(df, col_mapping)
        
        # 4. Detect Fraud
        df_results = detect_fraud(df_feat, col_mapping)
        
        # Calculate Metrics
        total_tx = len(df_results)
        fraud_tx = len(df_results[df_results['Prediction'] == 'Yes'])
        fraud_pct = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0
        
        # Determine Risk Classification
        if fraud_pct < 5:
            risk_label = "🟢 Low Risk"
        elif fraud_pct <= 15:
            risk_label = "🟡 Moderate Risk"
        else:
            risk_label = "🔴 High Risk"
            
        # --- UI DISPLAY ---
        
        st.success("✅ Analysis Complete!")
        
        st.header("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📝 Total Transactions", f"{total_tx:,}")
        col2.metric("🚨 Detected Frauds", f"{fraud_tx:,}")
        
        # Color code fraud percentage using HTML
        pct_color = "#4CAF50" if fraud_pct < 5 else "#FFC107" if fraud_pct <= 15 else "#F44336"
        col3.markdown(f"**📈 Fraud Percentage**<br><h2 style='color:{pct_color}; margin-top: -10px;'>{fraud_pct:.2f}%</h2>", unsafe_allow_html=True)
        col4.markdown(f"**⚠️ System Risk Level**<br><h2 style='margin-top: -10px;'>{risk_label}</h2>", unsafe_allow_html=True)
        
        st.divider()
        
        st.header("📈 Analytics & Insights")
        
        st.subheader("Distribution Analysis")
        st.markdown("*Overview of the transaction amounts and the proportion of flagged anomalies.*")
        c1, c2 = st.columns(2)
        
        with c1:
            fig_pie = plot_fraud_distribution(df_results)
            if fig_pie:
                st.pyplot(fig_pie, use_container_width=True)
                
        with c2:
            fig_hist = plot_amount_distribution(df_results, col_mapping)
            if fig_hist:
                st.pyplot(fig_hist, use_container_width=True)
                
        st.subheader("Behavior Insights")
        st.markdown("*Segmentation of users based on their spending behavior.*")
        fig_seg = plot_user_segmentation(df_results, col_mapping)
        if fig_seg:
            st.pyplot(fig_seg, use_container_width=True)
        else:
            st.info("User or Amount column not found for segmentation.")
            
        st.divider()
        
        st.header("💡 Top Fraud Indicators")
        st.markdown("""
        Based on the current analysis, the system primarily flags transactions exhibiting these patterns:
        - 🔴 **High Transaction Amount:** Values significantly above the 90th percentile.
        - 🌙 **Late Night Transactions:** Activity between 00:00 and 05:00.
        - 🚨 **Unusual Behavior:** Transactions falling significantly outside the user's norm (Anomaly Detection).
        """)
        
        st.divider()
        
        st.header("📋 Detailed Predictions")
        
        # --- Filtering & Sorting ---
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            filter_option = st.radio("Filter Predictions:", ["Show All", "Fraud Only", "Non-Fraud Only"], horizontal=True)
        with f_col2:
            sort_option = st.selectbox("Sort By:", ["Risk Score (High to Low)", "Risk Score (Low to High)", "Amount (High to Low)"])
            
        # Reorder columns to put Prediction and Risk Score first
        cols = df_results.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Prediction')))
        cols.insert(1, cols.pop(cols.index('Risk_Score')))
        df_display = df_results[cols]
        
        # Apply Filter
        if filter_option == "Fraud Only":
            df_display = df_display[df_display['Prediction'] == 'Yes']
        elif filter_option == "Non-Fraud Only":
            df_display = df_display[df_display['Prediction'] == 'No']
            
        # Apply Sort
        if sort_option == "Risk Score (High to Low)":
            df_display = df_display.sort_values(by='Risk_Score', ascending=False)
        elif sort_option == "Risk Score (Low to High)":
            df_display = df_display.sort_values(by='Risk_Score', ascending=True)
        elif "Amount" in sort_option and col_mapping.get('amount') and col_mapping.get('amount') in df_display.columns:
            df_display = df_display.sort_values(by=col_mapping.get('amount'), ascending=False)
        else:
            # Fallback if no amount column
            df_display = df_display.sort_values(by='Risk_Score', ascending=False)
        
        # Style the dataframe - Subtle highlight
        def highlight_fraud(row):
            if row['Prediction'] == 'Yes':
                return ['background-color: rgba(255, 0, 0, 0.15)'] * len(row)
            return [''] * len(row)
            
        st.dataframe(
            df_display.style.apply(highlight_fraud, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Provide download link
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv',
        )
        
        with st.expander("🔍 System Inferred Column Mapping"):
            st.json(col_mapping)
