# =================
# ==== IMPORTS ====
# =================

import pandas as pd
import streamlit as st

# ===================
# ==== FUNCTIONS ====
# ===================

# Load dataset
@st.cache
def _load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# Main function to run the app
def main():
    st.title("Velib Dataset Information")

    # Load data
    df_train = _load_data('data/04_feature/df_feat_train.parquet')

    # Display dataset
    st.subheader("Dataset")
    st.write(df_train)

    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(df_train.describe())

    # Display column names
    st.subheader("Column Names")
    st.write(df_train.columns)

    # Display data types
    st.subheader("Data Types")
    st.write(df_train.dtypes)

    # Display number of rows and columns
    st.subheader("Number of Rows and Columns")
    st.write(f"Rows: {df_train.shape[0]}, Columns: {df_train.shape[1]}")

if __name__ == "__main__":
    main()
