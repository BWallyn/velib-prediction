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
def load_data():
    df = pd.read_csv('path_to_your_velib_dataset.csv')
    return df

# Main function to run the app
def main():
    st.title("Velib Dataset Information")

    # Load data
    df = load_data()

    # Display dataset
    st.subheader("Dataset")
    st.write(df)

    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Display column names
    st.subheader("Column Names")
    st.write(df.columns)

    # Display data types
    st.subheader("Data Types")
    st.write(df.dtypes)

    # Display number of rows and columns
    st.subheader("Number of Rows and Columns")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

if __name__ == "__main__":
    main()
