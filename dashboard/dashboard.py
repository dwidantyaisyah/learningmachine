import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import streamlit as st

# Judul Dashboard
st.title("Proyek Analisis Data: [Bike Sharing Dataset")

# Load dataset (replace 'data.csv' with your dataset file)
uploaded_file = st.file_uploader("Upload Your Dataset", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Load data based on file type
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    # Ensure the date column is in datetime format
    date_column = st.selectbox("Select Date Column", data.columns)
    data[date_column] = pd.to_datetime(data[date_column])

    # Sidebar Filters
    st.sidebar.header("Filters")
    
    # Date Filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [data[date_column].min(), data[date_column].max()]
    )
    
    # Category Filter
    category_column = st.selectbox("Select Category Column", data.select_dtypes(include=['object', 'category']).columns)
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=data[category_column].unique(),
        default=data[category_column].unique()
    )
    
    # Apply filters
    filtered_data = data[
        (data[date_column] >= pd.to_datetime(date_range[0])) &
        (data[date_column] <= pd.to_datetime(date_range[1])) &
        (data[category_column].isin(selected_categories))
    ]
    
    st.write(f"Filtered Data ({len(filtered_data)} rows):")
    st.dataframe(filtered_data)

    # Visualizations
    st.header("Visualizations")

    # Line Chart
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    line_chart_column = st.selectbox("Select Column for Line Chart", numerical_columns)
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_data[date_column], filtered_data[line_chart_column], label=line_chart_column)
    plt.xlabel("Date")
    plt.ylabel(line_chart_column)
    plt.title(f"{line_chart_column} Over Time")
    plt.legend()
    st.pyplot(plt)

    # Bar Chart
    st.header("Bar Chart of Categories")
    bar_chart_data = filtered_data.groupby(category_column).size().reset_index(name='counts')
    plt.figure(figsize=(8, 5))
    sns.barplot(data=bar_chart_data, x=category_column, y='counts', palette='viridis')
    plt.title(f"Counts by {category_column}")
    plt.xlabel(category_column)
    plt.ylabel("Counts")
    st.pyplot(plt)

    # Correlation Heatmap
    st.header("Correlation Heatmap")
    corr = filtered_data.select_dtypes(include=['float64', 'int64']).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    st.pyplot(plt)

else:
    st.info("Please upload a dataset to begin.")
