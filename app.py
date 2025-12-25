import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from zipfile import ZipFile
import pickle
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime

from src.extract_features import FeaturesExtraction

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Segment cards */
    .segment-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Success message */
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Upload area */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Marketing strategies configuration
SEGMENT_STRATEGIES = {
    2: {
        'name': '🌟 VIP Champions',
        'emoji': '👑',
        'color': '#2ecc71',
        'description': 'Your most valuable customers - shop frequently and spend a lot',
        'characteristics': [
            '✅ Very high purchase rate (14+ times)',
            '✅ High spending value ($9,000+)',
            '✅ Last recent purchase (< 20 days)',
            '✅ High brand loyalty'
        ],
        'strategy': '🎯 Retention and Reward Strategy',
        'actions': [
            '💎 Exclusive VIP program with special privileges',
            '🎁 Personalized rewards and gifts for special occasions',
            '📞 Direct personal contact from customer management',
            '⚡ Early access to new products',
            '🎫 Invitations to exclusive events',
            '💳 Exclusive discounts 15-25%'
        ],
        'roi': 'ROI:🚀 very high',
        'budget': 'Suggested budget: 30-40% of the marketing budget'
    },
    3: {
        'name': '✨ Potential Loyalists',
        'emoji': '⭐',
        'color': '#3498db',
        'description': 'Regular customers with the possibility of being upgraded to VIP',
        'characteristics': [
            '✅ Good purchase rate (4-5 times)',
            '✅ Average spending value ($2,000)',
            '✅ Recent activity (< 35 days)',
            '✅ High growth potential'
        ],
        'strategy': '🎯 Development and Additional Sales Strategy',
        'actions': [
            '🎯 Targeted promotional offers based on purchasing behavior',
            '🏆 Progressive loyalty points program',
            '📧 Smart product recommendations via email',
            '🔔 Alerts for products similar to their purchases',
            '💰 Product Bundle Offers Discounts',
            '📱 Follow-up messages after purchase'
        ],
        'roi': 'ROI: High 📈',
        'budget': 'Suggested budget: 25-30% of the marketing budget'
    },
    0: {
        'name': '📊 Average Customers',
        'emoji': '⚖️',
        'color': '#f39c12',
        'description': 'Average customers need growth incentives',
        'characteristics': [
            '⚡ Low purchase rate (2-3 times)',
            '⚡ Average spending value ($1,700)',
            '⚡ Last average purchase (85 days)',
            '⚡ They need to activate'
        ],
        'strategy': '🎯 Activation and Motivation Strategy',
        'actions': [
            '🎁 Incentive offers for repeat purchases',
            '📱 Targeted SMS/WhatsApp campaigns',
            '🎯 Limited-time discount coupons (10-15%)',
            '📧 Newsletter with offers',
            '🎮 Interactive competitions and games',
            '⭐ "Buy X, Get Y" program'
        ],
        'roi': 'ROI: Average 📊',
        'budget': 'Suggested budget: 20-25% of the marketing budget'
    },
    1: {
        'name': '⚠️ At Risk',
        'emoji': '🔔',
        'color': '#e67e22',
        'description': 'Customers showing signs of disinterest',
        'characteristics': [
            '⚠️ Very low purchase rate (1-2 times)',
            '⚠️ Low spending value ($374)',
            '⚠️ Last old purchase (145 days ago)',
            '⚠️ The risk of losing them is high.'
        ],
        'strategy': '🎯 Re-engagement and Retrieval Strategy',
        'actions': [
            '🎁 "We Miss You" Campaigns (Win-back Campaigns)',
            '💰 Huge discounts of 20-30% to attract new customers',
            '📞 Personal calls to inquire about their experience',
            '📋 Customer satisfaction surveys',
            '🆓 Free samples or free shipping',
            '🎯 Re-targeting ads on social media'
        ],
        'roi': 'ROI: Medium-low 📉',
        'budget': 'Suggested budget: 10-15% of the marketing budget'
    },
    4: {
        'name': '❌ Lost/Hibernating',
        'emoji': '😴',
        'color': '#e74c3c',
        'description': 'Inactive customers - about to be lost',
        'characteristics': [
            '❌ Very low purchase rate (1 time)',
            '❌ Low spending value ($170)',
            '❌ Last purchase is very old (163+ days)',
            '❌ Low probability of return'
        ],
        'strategy': '🎯 Last resort or ignore it Strategy',
        'actions': [
            '🎯 The "Last Chance" campaign with an emotional message',
            '💥 Huge discounts 40-50% (Last Chance)',
            '📧 Email reminder for products in cart',
            '🔍 Analysis of the causes of the stoppage (Exit Survey)',
            '🎁 Exceptional Comeback Offer',
            '⚖️ Evaluate: Are they worth the investment?'
        ],
        'roi': 'ROI: Low ⚠️',
        'budget': 'Suggested budget: 5-10% of the marketing budget'
    }
}


class CustomerSegmentationApp:
    """Main application class for customer segmentation"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained model and scaler"""
        try:
            with open('artifacts/k_means_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('artifacts/robust_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            logging.info("Models loaded successfully")
        except FileNotFoundError:
            st.error("⚠️ Model files not found. Please train the model first.")
            logging.error("Model files not found")
    
    def validate_input_file(self, df):
        """Validate uploaded file structure"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        required_columns = [
            'InvoiceDate', 'InvoiceID', 'CustomerID','ProductID', 
            'Quantity', 'UnitPrice']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        return True, "Validation passed"
    
    def preprocess_data(self, df):
        """Preprocess input data"""
        # Drop Duplicates
        df = df.drop_duplicates()
        # Drop Missing Values
        df = df.dropna()
        # Keep Date only in `InvoiceDate` Column
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df['InvoiceDate'] = df['InvoiceDate'].dt.normalize()
        # Rename Some Columns
        df.rename(columns={
            "InvoiceID": "InvoiceNo", "ProductID": "StockCode"
            }, inplace=True)
        # Extract Features
        data_extractor = FeaturesExtraction()
        extracted_df = data_extractor.extract(df)
        
        # Apply log transformation
        numeric_cols = ['Recency', 'Frequency', 'Monetary', 'Avg_Basket_Size', 'CLV']
        transformed_df = extracted_df.copy()
        
        transformed_df = transformed_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        for col in numeric_cols:
            transformed_df[col] = np.log1p(transformed_df[col])
        
        # Scale the data
        customer_ids = transformed_df['CustomerID']
        features = transformed_df.drop(columns=['CustomerID'])
        scaled_data = self.scaler.transform(features)
        
        extracted_df = extracted_df[extracted_df['CustomerID'].isin(customer_ids)]
        
        return scaled_data, customer_ids, extracted_df
    
    def predict_segments(self, scaled_data):
        """Predict customer segments"""
        predictions = self.model.predict(scaled_data)
        return predictions
    
    def create_results_dataframe(self, original_df, predictions):
        """Create comprehensive results dataframe"""
        results_df = original_df.copy()
        results_df['Segment'] = predictions
        results_df['Segment_Name'] = results_df['Segment'].map(
            lambda x: SEGMENT_STRATEGIES[x]['name']
        )
        results_df['Priority'] = results_df['Segment'].map({
            2: '🔥 Highest',
            3: '⭐ High',
            0: '📊 Medium',
            1: '⚠️ Low',
            4: '❌ Very Low'
        })
        return results_df


def render_header():
    """Render application header"""
    st.markdown("""
        <div class="main-header">
            <h1>🛒 AI-Powered Customer Segmentation Platform</h1>
            <p style="font-size: 1.2rem; margin-top: 1rem;">
                Transform your customer data into actionable marketing strategies
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with information"""
    with st.sidebar:
        st.image("demo/customer_segmentation.jpg", width=120)
        st.title("📊 About")
        
        st.markdown("""
        ### How It Works
        
        1️⃣ **Upload** your customer data file (CSV/Excel/ZIP/SQL)
        
        2️⃣ **AI Analysis** segments customers automatically
        
        3️⃣ **Get Insights** with detailed marketing strategies
        
        4️⃣ **Take Action** based on recommendations
        
        ---
        
        ### Required Data Format
        
        Your file must include:
        - `CustomerID`: Unique identifier
        - `InvoiceID`: Unique identifier
        - `ProductID`: Unique identifier
        - `InvoiceData`: Invoice Date
        - `Quantity`: Quantity of product
        - `UnitPrice`: Unit Price
        
        ---
        
        ### 📥 Sample Data
        """)
        
        # Sample data download
        sample_data = pd.DataFrame({
            "InvoiceDate": ["2024-11-28", "2024-11-28", "2024-11-28", "2024-11-15", "2024-11-15",
                "2024-10-20", "2024-10-20", "2024-10-20", "2024-09-25", "2024-09-10",
                "2024-11-30", "2024-11-30", "2024-11-10", "2024-11-10", "2024-11-10",
                "2024-10-15", "2024-10-15", "2024-09-20", "2024-11-05", "2024-11-05",
                "2024-09-15", "2024-09-15", "2024-07-20", "2024-07-20", "2024-05-10",
                "2024-10-25", "2024-10-25", "2024-08-30", "2024-08-30", "2024-06-15", "2024-06-15",
                "2024-04-20", "2024-11-12", "2024-11-12", "2024-09-08", "2024-07-15", "2024-07-15",
                "2024-05-25", "2024-10-10", "2024-07-25", "2024-05-05", "2024-03-12",
                "2024-09-18", "2024-06-20", "2024-04-15", "2024-08-22", "2024-05-30", "2024-02-14",
                "2024-06-10", "2024-02-20", "2024-05-15", "2024-01-10"],
            
            "InvoiceID": ["INV50001", "INV50001", "INV50001", "INV50002", "INV50002",
                "INV50003", "INV50003", "INV50003", "INV50004", "INV50005","INV50006", "INV50006",
                "INV50007", "INV50007", "INV50007", "INV50008", "INV50008", "INV50009",
                "INV50010", "INV50010", "INV50011", "INV50011", "INV50012", "INV50012","INV50013",
                "INV50014", "INV50014", "INV50015", "INV50015", "INV50016", "INV50016", "INV50017",
                "INV50018", "INV50018", "INV50019", "INV50020", "INV50020", "INV50021",
                "INV50022", "INV50023", "INV50024", "INV50025", "INV50026", "INV50027", "INV50028",
                "INV50029", "INV50030", "INV50031", "INV50032", "INV50033", "INV50034", "INV50035"
            ],
            
            "CustomerID": ["C10001", "C10001", "C10001", "C10001", "C10001", "C10001", "C10001", "C10001",
                "C10001", "C10001", "C10002", "C10002", "C10002", "C10002", "C10002", "C10002", "C10002",
                "C10002", "C10003", "C10003", "C10003", "C10003", "C10003", "C10003", "C10003",
                "C10004", "C10004", "C10004", "C10004", "C10004", "C10004", "C10004",
                "C10005", "C10005", "C10005", "C10005", "C10005", "C10005",
                "C10006", "C10006", "C10006", "C10006", "C10007", "C10007", "C10007", "C10008", "C10008",
                "C10008", "C10009", "C10009", "C10010", "C10010"
            ],
            
            "ProductID": [ "P001", "P005", "P011", "P007", "P008", "P005", "P011", "P012", "P001", "P007",
                "P001", "P004", "P005", "P008", "P011", "P007", "P012", "P002", "P002", "P006", "P004", "P007",
                "P003", "P009", "P006", "P003", "P010", "P006", "P009", "P002", "P007", "P004",
                "P004", "P008", "P006", "P003", "P009", "P007", "P002", "P006", "P003", "P010", "P002",
                "P006", "P009", "P003", "P010", "P002", "P002", "P006", "P003", "P002"
            ],
            
            "Quantity": [3, 2, 4, 2, 3, 2, 3, 2, 4, 2, 3, 2, 2, 3, 2, 3, 2, 4, 2, 3, 2, 2, 2, 1, 3,
                2, 1, 3, 2, 2, 2, 1, 2, 2, 3, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1
            ],
            
            "UnitPrice": [39.99, 55.00, 65.00, 24.99, 38.50, 52.00, 68.00, 32.50, 42.00, 26.50,
                41.50, 29.99, 58.00, 36.00, 62.50, 25.50, 34.00, 12.99, 13.50, 18.00, 28.50, 22.00,
                19.99, 16.50, 17.50, 21.00, 14.99, 19.50, 17.00, 11.99, 23.50, 27.00, 30.50, 35.00,
                18.50, 20.00, 15.50, 24.00, 9.99, 16.50, 18.00, 12.50, 11.50, 17.00, 14.99, 19.50,
                13.00, 10.50, 8.99, 15.50, 17.50, 9.50
            ]
        })
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Sample Template",
            data=csv,
            file_name="customer_data_template.csv",
            mime="text/csv",
            help="Download this template to see the required format"
        )
        
        st.markdown("---")
        st.markdown("""
        ### 🎓 Developed By
        **Emran Albeik**  
        ML Engineer | Data Analyst
        
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <div style="display: flex; gap: 20px; font-size: 28px;">
            <a href="http://linkedin.com/in/emranalbeik" target="_blank"><i class="fab fa-linkedin"></i></a>
            <a href="https://github.com/RedDragon30" target="_blank"><i class="fab fa-github"></i></a>
            <a href="https://emranalbeik.odoo.com/" target="_blank"><i class="fas fa-globe"></i></a>
            <a href="mailto:emranalbiek@gmail.com"><i class="fas fa-envelope"></i></a>
        </div>
    """, unsafe_allow_html=True)


def render_upload_section(app):
    """Render file upload section"""
    st.markdown("## 📁 Upload Customer Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a `CSV` or `Excel` or `ZIP` or `SQL Database` file",
            type=['csv', 'xlsx', 'xls', 'zip', 'db'],
            help="Upload your customer data file with required columns"
        )
    
    with col2:
        st.info("""
        **Supported formats:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - ZIP (.zip)
        - SQL (.db)
        
        **Max size:** 200MB
        """)
    
    return uploaded_file

def render_overview_metrics(results_df):
    """Render overview metrics"""
    st.markdown("## 📊 Customer Base Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(results_df)
    avg_monetary = results_df['Monetary'].mean()
    avg_frequency = results_df['Frequency'].mean()
    total_revenue = results_df['Monetary'].sum()
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}",
            delta="Analyzed"
        )
    
    with col2:
        st.metric(
            label="Avg Customer Value",
            value=f"${avg_monetary:,.0f}",
            delta=f"{(avg_monetary/total_revenue*100):.1f}% of total"
        )
    
    with col3:
        st.metric(
            label="Avg Purchase Frequency",
            value=f"{avg_frequency:.1f}x",
            delta="Per customer"
        )
    
    with col4:
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta="Lifetime value"
        )


def render_segment_distribution(results_df):
    """Render segment distribution visualizations"""
    st.markdown("## 🎯 Segment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        segment_counts = results_df['Segment_Name'].value_counts()
        colors = [SEGMENT_STRATEGIES[seg]['color'] for seg in results_df['Segment'].unique()]
        
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Customer Distribution by Segment",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart with monetary value
        segment_value = results_df.groupby('Segment_Name')['Monetary'].sum().sort_values(ascending=True)
        
        fig = go.Figure(data=[go.Bar(
            y=segment_value.index,
            x=segment_value.values,
            orientation='h',
            marker=dict(
                color=segment_value.values,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"${val:,.0f}" for val in segment_value.values],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Total Revenue by Segment",
            xaxis_title="Total Revenue ($)",
            yaxis_title="Segment",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_segment_analysis(results_df):
    """Render detailed segment analysis"""
    st.markdown("## 📈 Segment Performance Analysis")
    
    # Create metrics table
    segment_metrics = results_df.groupby('Segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum'],
        'CLV': 'mean'
    }).round(2)
    
    segment_metrics.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 
                                'Avg_Monetary', 'Total_Revenue', 'Avg_CLV']
    segment_metrics['Revenue_Share_%'] = (
        segment_metrics['Total_Revenue'] / segment_metrics['Total_Revenue'].sum() * 100
    ).round(1)
    
    # Add segment names
    segment_metrics['Segment_Name'] = segment_metrics.index.map(
        lambda x: SEGMENT_STRATEGIES[x]['name']
    )
    
    # Reorder columns
    segment_metrics = segment_metrics[['Segment_Name', 'Count', 'Avg_Recency', 
                                    'Avg_Frequency', 'Avg_Monetary', 
                                    'Total_Revenue', 'Revenue_Share_%', 'Avg_CLV']]
    
    st.dataframe(
        segment_metrics.style.background_gradient(cmap='RdYlGn', subset=['Avg_Frequency', 'Avg_Monetary'])
                            .format({
                                'Avg_Recency': '{:.0f} days',
                                'Avg_Frequency': '{:.1f}x',
                                'Avg_Monetary': '${:,.0f}',
                                'Total_Revenue': '${:,.0f}',
                                'Revenue_Share_%': '{:.1f}%',
                                'Avg_CLV': '${:,.0f}'
                            }),
        use_container_width=True,
        height=300
    )


def render_rfm_analysis(results_df):
    """Render RFM analysis visualization"""
    st.markdown("## 🔍 RFM Analysis (3D Visualization)")
    
    fig = go.Figure(data=[go.Scatter3d(
        x=results_df['Recency'],
        y=results_df['Frequency'],
        z=results_df['Monetary'],
        mode='markers',
        marker=dict(
            size=5,
            color=results_df['Segment'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Segment"),
            opacity=0.8
        ),
        text=results_df['Segment_Name'],
        hovertemplate='<b>%{text}</b><br>' +
                    'Recency: %{x:.0f} days<br>' +
                    'Frequency: %{y:.0f}x<br>' +
                    'Monetary: $%{z:,.0f}<br>' +
                    '<extra></extra>'
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Recency (days)',
            yaxis_title='Frequency',
            zaxis_title='Monetary ($)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=600,
        title="3D RFM Scatter Plot"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_marketing_strategies():
    """Render marketing strategies for each segment"""
    st.markdown("## 🎯 Marketing Strategies & Action Plans")
    
    st.info("💡 **Pro Tip:** Implement these strategies in order of segment priority for maximum ROI")
    
    # Sort segments by priority (VIP first)
    priority_order = [2, 3, 0, 1, 4]
    
    for segment_id in priority_order:
        strategy = SEGMENT_STRATEGIES[segment_id]
        
        with st.expander(f"{strategy['emoji']} {strategy['name']} - {strategy['strategy']}", 
                        expanded=(segment_id == 2)):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### 📋 Description")
                st.write(strategy['description'])
                
                st.markdown(f"### 🎯 Key Characteristics")
                for char in strategy['characteristics']:
                    st.markdown(f"- {char}")
                
                st.markdown(f"### 🚀 Recommended Actions")
                for action in strategy['actions']:
                    st.markdown(f"- {action}")
            
            with col2:
                st.markdown(f"### 💰 Investment")
                st.success(strategy['roi'])
                st.info(strategy['budget'])
                
                # Add engagement score visualization
                engagement_score = {2: 95, 3: 75, 0: 50, 1: 30, 4: 10}[segment_id]
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=engagement_score,
                    title={'text': "Engagement Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': strategy['color']},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "lightblue"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")


def render_customer_details(results_df):
    """Render individual customer details"""
    st.markdown("## 👥 Customer Details")
    
    # Search and filter options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_customer = st.text_input("🔍 Search Customer ID", "")
    
    with col2:
        segment_filter = st.multiselect(
            "Filter by Segment",
            options=results_df['Segment_Name'].unique(),
            default=results_df['Segment_Name'].unique()
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=['Monetary', 'Frequency', 'Recency', 'CLV'],
            index=0
        )
    
    # Filter data
    filtered_df = results_df[results_df['Segment_Name'].isin(segment_filter)]
    
    if search_customer:
        filtered_df = filtered_df[
            filtered_df['CustomerID'].str.contains(search_customer, case=False, na=False)
        ]
    
    # Sort data
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
    
    # Display results
    st.dataframe(
        filtered_df[['CustomerID', 'Segment_Name', 'Priority', 'Recency', 
                    'Frequency', 'Monetary', 'Avg_Basket_Size', 'CLV']]
        .style.apply(lambda x: ['background-color: #d4edda' if x.name in filtered_df.index[:10] else '' 
                                for i in x], axis=1),
        use_container_width=True,
        height=400
    )
    
    st.caption(f"📊 Showing {len(filtered_df)} of {len(results_df)} customers")


# Business Impact Calculator
def render_roi_calculator(results_df):
    st.markdown("## 💰 ROI Impact Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_budget = st.number_input(
            "Monthly Marketing Budget ($)", 
            min_value=1000, 
            value=10000, 
            step=1000
        )
        
        conversion_rate = st.slider(
            "Expected Conversion Rate Increase (%)", 
            min_value=1, 
            max_value=50, 
            value=15
        )
    
    with col2:
        # Calculate potential revenue increase
        vip_customers = len(results_df[results_df['Segment'] == 2])
        avg_vip_value = results_df[results_df['Segment'] == 2]['Monetary'].mean()
        
        potential_revenue = vip_customers * avg_vip_value * (conversion_rate/100)
        roi = (potential_revenue / monthly_budget) * 100
        
        st.metric("Potential Revenue Increase", f"${potential_revenue:,.0f}")
        st.metric("Expected ROI", f"{roi:.1f}%")
        st.success(f"💡 Focus on VIP segment could generate ${potential_revenue:,.0f} additional revenue")

# Predictive Analytics
def render_churn_prediction(results_df):
    st.markdown("## 🔮 Churn Risk Prediction")
    
    # Calculate churn risk score
    results_df['Churn_Risk'] = (
        (results_df['Recency'] / results_df['Recency'].max() * 40) +
        (1 - results_df['Frequency'] / results_df['Frequency'].max()) * 30 +
        (1 - results_df['Monetary'] / results_df['Monetary'].max()) * 30
    )
    
    high_risk = len(results_df[results_df['Churn_Risk'] > 70])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Risk Customers", high_risk, "⚠️ Action needed")
    
    with col2:
        potential_loss = results_df[results_df['Churn_Risk'] > 70]['CLV'].sum()
        st.metric("Potential Revenue Loss", f"${potential_loss:,.0f}", "❌ At risk")
    
    with col3:
        retention_budget = potential_loss * 0.15
        st.metric("Recommended Retention Budget", f"${retention_budget:,.0f}", "💰 Invest")
    
    # Visualization
    fig = px.histogram(results_df, x='Churn_Risk', nbins=20, 
                    title="Churn Risk Distribution",
                    color_discrete_sequence=['#e74c3c'])
    st.plotly_chart(fig, use_container_width=True)

def render_export_section(results_df):
    """Render export options"""
    st.markdown("## 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export full results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Full Report (CSV)",
            data=csv,
            file_name=f"customer_segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download complete analysis with all customer details"
        )
    
    with col2:
        # Export segment summary
        segment_summary = results_df.groupby('Segment_Name').agg({
            'CustomerID': 'count',
            'Monetary': 'sum',
            'Frequency': 'mean'
        }).reset_index()
        
        summary_csv = segment_summary.to_csv(index=False)
        st.download_button(
            label="📈 Download Summary (CSV)",
            data=summary_csv,
            file_name=f"segment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download segment-level summary statistics"
        )
    
    with col3:
        # Export action plan
        action_plan = []
        for seg_id, strategy in SEGMENT_STRATEGIES.items():
            action_plan.append({
                'Segment': strategy['name'],
                'Strategy': strategy['strategy'],
                'Priority': ['Highest', 'High', 'Medium', 'Low', 'Very Low'][
                    [2, 3, 0, 1, 4].index(seg_id)
                ],
                'Actions': ' | '.join(strategy['actions'][:3])
            })
        
        action_df = pd.DataFrame(action_plan)
        action_csv = action_df.to_csv(index=False)
        
        st.download_button(
            label="🎯 Download Action Plan (CSV)",
            data=action_csv,
            file_name=f"marketing_action_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download marketing strategies and action items"
        )

def main():
    """Main application function"""
    
    # Initialize app
    app = CustomerSegmentationApp()
    
    # Render UI components
    render_header()
    render_sidebar()
    
    # File upload section
    uploaded_file = render_upload_section(app)
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            
            elif uploaded_file.name.endswith('.zip'):
                with ZipFile(uploaded_file, "r") as z:
                    xls_files = [f for f in z.namelist() if f.endswith('.xlsx')]
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    
                    if len(xls_files) != 0:
                        with z.open(xls_files[0]) as f:
                            df = pd.read_excel(f)
                    elif len(csv_files) != 0:
                        with z.open(csv_files[0]) as f:
                            df = pd.read_csv(f)
                    else:
                        raise TypeError("Not Supported File")
            
            elif uploaded_file.name.endswith('.db'):
                conn = sqlite3.connect(uploaded_file)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                table_names = [name[0] for name in cursor.fetchall()]
                selected_table = st.selectbox("Select Table to Analyze", table_names)
                if selected_table:
                    df = pd.read_sql(f"SELECT * FROM {selected_table};", conn)
                else:
                    st.error("No tables found in the database.")
                    st.stop()
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate input
            is_valid, message = app.validate_input_file(df)
            
            if not is_valid:
                st.error(f"❌ Validation Error: {message}")
                st.stop()
            
            st.success(f"✅ File uploaded successfully! {len(df)} records detected.")
            
            # Process data
            with st.spinner("🔄 Processing data and generating insights..."):
                scaled_data, customer_ids, original_df = app.preprocess_data(df)
                predictions = app.predict_segments(scaled_data)
                results_df = app.create_results_dataframe(original_df, predictions)
            
            st.success("✨ Analysis completed successfully!")
            
            # Render results sections
            st.markdown("---")
            render_overview_metrics(results_df)
            
            st.markdown("---")
            render_segment_distribution(results_df)
            
            st.markdown("---")
            render_segment_analysis(results_df)
            
            st.markdown("---")
            render_rfm_analysis(results_df)
            
            st.markdown("---")
            render_marketing_strategies()
            
            st.markdown("---")
            render_customer_details(results_df)
            
            st.markdown("---")
            render_roi_calculator(results_df)
            
            st.markdown("---")
            render_churn_prediction(results_df)
            
            st.markdown("---")
            render_export_section(results_df)
            
            # Footer
            st.markdown("---")
            st.markdown("""
                <div style="text-align: center; color: #666; padding: 2rem;">
                    <p>Made with ❤️ using Streamlit | © 2025 Customer Segmentation AI</p>
                    <p>Powered by AI</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logging.exception("Application error")
            
            with st.expander("🔍 View Error Details"):
                st.code(str(e))
    
    else:
        # Landing page when no file is uploaded
        st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; margin: 2rem 0;">
                <h2>👋 Welcome to Customer Segmentation AI</h2>
                <p style="font-size: 1.2rem; margin: 1rem 0;">
                    Upload your customer data to get started with intelligent segmentation and marketing strategies
                </p>
                <p style="color: #666;">
                    📁 Drag and drop your file above or click to browse
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Benefits section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                ### 🎯 Smart Segmentation
                AI-powered clustering identifies 5 distinct customer segments based on purchasing behavior
            """)
        
        with col2:
            st.markdown("""
                ### 📊 Actionable Insights
                Get detailed marketing strategies tailored for each customer segment
            """)
        
        with col3:
            st.markdown("""
                ### 💰 Maximize ROI
                Optimize marketing spend by targeting the right customers with the right strategy
            """)


if __name__ == "__main__":
    main()
