import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

# Load cleaned data
all_df = pd.read_csv("./dashboard/main_data.csv")

datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(drop=True, inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Calculate delivery time
all_df["delivery_time"] = (
    all_df["order_delivered_customer_date"] - all_df["order_purchase_timestamp"]
).dt.days

# Function to create RFM dataframe
def create_rfm_df(all_df):
    reference_date = all_df['order_purchase_timestamp'].max()
    rfm_df = all_df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).reset_index()

    rfm_df.rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'price': 'Monetary'
    }, inplace=True)

    rfm_df['r_rank'] = rfm_df['Recency'].rank(ascending=False)
    rfm_df['f_rank'] = rfm_df['Frequency'].rank(ascending=True)
    rfm_df['m_rank'] = rfm_df['Monetary'].rank(ascending=True)

    rfm_df['r_rank_norm'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max()) * 100
    rfm_df['f_rank_norm'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max()) * 100
    rfm_df['m_rank_norm'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max()) * 100

    rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

    rfm_df['RFM_score'] = 0.15 * rfm_df['r_rank_norm'] + \
                          0.28 * rfm_df['f_rank_norm'] + \
                          0.57 * rfm_df['m_rank_norm']

    rfm_df['RFM_score'] *= 0.05
    rfm_df = rfm_df.round(2)

    rfm_df["customer_segment"] = np.where(
        rfm_df['RFM_score'] > 4.5, "Top customers", np.where(
            rfm_df['RFM_score'] > 4, "High value customer", np.where(
                rfm_df['RFM_score'] > 3, "Medium value customer", np.where(
                    rfm_df['RFM_score'] > 1.6, 'Low value customers', 'Lost customers'))))

    customer_segment_df = rfm_df.groupby(by="customer_segment", as_index=False).customer_unique_id.nunique()
    return rfm_df, customer_segment_df

# Sidebar filters
with st.sidebar:
    st.title("Filters")
    try:
        min_date = all_df["order_purchase_timestamp"].min()
        max_date = all_df["order_purchase_timestamp"].max()

        start_date, end_date = st.date_input(
            "Rentang Waktu",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
    except ValueError:
        st.error("Masukkan tanggal mulai dan akhir yang valid.")
        start_date, end_date = min_date, max_date

main_df = all_df[
    (all_df["order_purchase_timestamp"] >= pd.Timestamp(start_date)) &
    (all_df["order_purchase_timestamp"] <= pd.Timestamp(end_date))
]

# Visualizations
def display_visualisasi_pertama(df):
    st.header("Distribusi Frekuensi Pesanan Berdasarkan Status")
    status_counts = df["order_status"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(status_counts.index, status_counts.values, color="skyblue")
    ax.set_title("Distribusi Frekuensi Pesanan")
    ax.set_xlabel("Status Pesanan")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

def display_visualisasi_kedua(df):
    st.header("Distribusi Waktu Pengiriman")
    delivery_time = df["delivery_time"]
    fig, ax = plt.subplots()
    ax.hist(delivery_time, bins=20, color="skyblue", edgecolor="black")
    ax.set_title("Distribusi Waktu Pengiriman")
    ax.set_xlabel("Waktu (hari)")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

def display_rfm(rfm_df):
    st.header("RFM Analysis")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    sns.barplot(data=rfm_df.sort_values(by="Recency").head(5), x="customer_unique_id", y="Recency", ax=ax[0])
    ax[0].set_title("Recency")

    sns.barplot(data=rfm_df.sort_values(by="Frequency", ascending=False).head(5), x="customer_unique_id", y="Frequency", ax=ax[1])
    ax[1].set_title("Frequency")

    sns.barplot(data=rfm_df.sort_values(by="Monetary", ascending=False).head(5), x="customer_unique_id", y="Monetary", ax=ax[2])
    ax[2].set_title("Monetary")

    st.pyplot(fig)

def display_customer_segment(customer_segment_df):
    st.header("Customer Segments")
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(data=customer_segment_df, x="customer_segment", y="customer_unique_id", palette="viridis")
    plt.title("Number of Customers by Segment")
    st.pyplot(fig)

# Generate RFM analysis
rfm_df, customer_segment_df = create_rfm_df(main_df)

st.title("E-Commerce Data Analysis Dashboard")
display_visualisasi_pertama(main_df)
display_visualisasi_kedua(main_df)
display_rfm(rfm_df)
display_customer_segment(customer_segment_df)
