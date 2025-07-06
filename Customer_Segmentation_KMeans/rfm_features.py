# rfm_features.py
import pandas as pd
import datetime as dt

def load_and_prepare_data(path):
    df = pd.read_excel(path)
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    return df

def compute_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    })
    rfm.columns = ['Recency', 'Frequency', 'Quantity', 'AvgPrice']

    rfm['Monetary'] = df.groupby('CustomerID').apply(lambda x: (x.Quantity * x.UnitPrice).sum())
    rfm = rfm[['Recency', 'Frequency', 'Monetary']]  # Keep important ones only
    return rfm
