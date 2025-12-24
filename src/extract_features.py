import logging
import pandas as pd

# Create Class for Features Extraction
class FeaturesExtraction():
    def extract(self,  df:pd.DataFrame) -> pd.DataFrame:
        """Extracts new features from the dataframe"""
        try:
            # Verify the presence of the required columns
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            required_cols = ['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'InvoiceDate', 'CustomerID']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing Columns: {missing_cols}")
            
            # 1) is_return (Record the return invoices)
            df['is_return'] = (df['Quantity'] < 0) | (df['InvoiceNo'].astype(str).str.startswith('C'))
            logging.info(f'Total transactions: {len(df)}')
            logging.info(f'Return transactions: {df['is_return'].sum()} ({df["is_return"].mean()*100:.2f}%)')
            
            # 2) Revenue
            df['Revenue'] = df['UnitPrice'] * df['Quantity']
            df = df[~(df['Revenue'] < 0)]
            
            # Remove returns transactions
            df = df[~df['is_return']]
            df = df[df['Quantity'] > 0]
            
            # 3) Recency (Calculate the difference in days between snapshot and customer last purchase date)
            snapshot_date= df['InvoiceDate'].max() + pd.Timedelta(days=1)
            recency = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
            recency['Recency'] = (snapshot_date - recency['InvoiceDate']).dt.days
            df = df.merge(recency[['CustomerID', 'Recency']], on='CustomerID')
            
            # 4) Frequency (Number of unique invoices per customer)
            freq = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
            freq.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)
            df = df.merge(freq, on='CustomerID')
            
            # 5) Monetary (Sum of Revenue pre customer)
            monetary= df.groupby('CustomerID')['Revenue'].sum().reset_index()
            monetary.rename(columns={'Revenue': 'Monetary'}, inplace=True)
            df = df.merge(monetary, on='CustomerID')
            
            # 6) Average Basket Size (Divides total products on total unique invoices per customer)
            basket_size = df.groupby('CustomerID')['Quantity'].sum().reset_index()
            basket_size['Avg_Basket_Size'] = basket_size['Quantity'] / df.groupby('CustomerID')['Frequency'].first().values
            df = df.merge(basket_size[['CustomerID', 'Avg_Basket_Size']], on='CustomerID')
            
            # Simple Features Creation
            # customers_features = df.groupby('CustomerID').agg({
            #     'InvoiceDate': 'max', # for Recency
            #     'InvoiceNo': 'nunique' # for Frequency
            #     'Revenue': 'sum', # for Monetary
            #     'Quantity': 'sum', # for Basket Size
            #     'StockCode': 'nunique', # for Diversity
            # }).reset_index()
            
            
            # 7) CLV (Customer Lifetime Value)
            # Step 1:Calculate AOV (Average Order Value)
            customer_metrics = df.groupby('CustomerID').agg({
                'Monetary': 'first',
                'Frequency': 'first'
            }).reset_index()
            
            customer_metrics['AOV'] = customer_metrics['Monetary'] / customer_metrics['Frequency']
            
            # Step 2:Calculate Customer Lifespan
            customer_lifetime = df.groupby('CustomerID').agg(
            first_purchase=('InvoiceDate', 'min'),
            last_purchase=('InvoiceDate', 'max')
        ).reset_index()
            
            customer_lifetime['lifespan_days'] = (
                customer_lifetime['last_purchase'] -
                customer_lifetime['first_purchase']
        ).dt.days
            
            customer_lifetime['lifespan_days'] = customer_lifetime['lifespan_days'].apply(
            lambda x: max(x, 30)
            )
            
            customer_lifetime['lifespan_years'] = customer_lifetime['lifespan_days'].mean() / 365
            
            customer_metrics = customer_metrics.merge(
                customer_lifetime[['CustomerID', 'lifespan_years']], 
                on='CustomerID'
            )
            
            customer_metrics['CLV'] = (
                customer_metrics['AOV'] * 
                customer_metrics['Frequency'] * 
                customer_metrics['lifespan_years']
            )
            
            df = df.merge(
                customer_metrics[['CustomerID', 'CLV']], 
                on='CustomerID', 
                how='left'
            )
            
            # Drop Useless Columns
            useless_cols = ['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'InvoiceDate', 'is_return', 'Revenue']
            df = df.drop(columns=useless_cols)
            
            # Drop Duplicates
            df = df.drop_duplicates(subset=['CustomerID'])
            
            logging.info('Features Extraction Completed')
            return df
        except Exception as e:
            logging.error(f'Error in features extraction: {e}')
            raise e