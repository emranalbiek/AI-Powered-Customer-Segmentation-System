# 🛒 AI-Powered Customer Segmentation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-Apache2.0-green)

**Transform your customer data into actionable marketing strategies using AI-powered segmentation**

</div>

## 🎯 Overview

The **Customer Segmentation System** is an end-to-end machine learning solution that analyzes customer purchasing behavior and automatically segments customers into distinct groups. The system provides actionable marketing strategies for each segment, helping businesses optimize their marketing spend and maximize ROI.

### 🎪 Key Highlights

- **5 Customer Segments**: VIP Champions, Potential Loyalists, Average Customers, At Risk, and Lost/Hibernating
- **RFM Analysis**: Recency, Frequency, Monetary value + CLV (Customer Lifetime Value)
- **3 ML Algorithms**: K-Means, Agglomerative Clustering, Gaussian Mixture Models
- **Automated Hyperparameter Tuning**: Using Optuna for optimal performance
- **Interactive Dashboard**: Beautiful Streamlit web application
- **Marketing Playbook**: Pre-defined strategies for each customer segment

---

## ✨ Features

### 🤖 Machine Learning Capabilities

- **Multiple Clustering Algorithms**

  - K-Means Clustering
  - Agglomerative Hierarchical Clustering
  - Gaussian Mixture Models (GMM)

- **Advanced Feature Engineering**

  - RFM (Recency, Frequency, Monetary) Analysis
  - Customer Lifetime Value (CLV) Calculation
  - Average Basket Size
  - Log Transformation
  - Robust Scaling

- **Automated Hyperparameter Tuning**
  - Optuna-based optimization
  - Configurable trial numbers
  - Silhouette score maximization

### 📊 Analytics & Visualization

- **Interactive 3D RFM Scatter Plots**
- **Segment Distribution Charts** (Pie & Bar)
- **Performance Metrics Dashboard**
- **Churn Risk Prediction**
- **ROI Impact Calculator**

### 🎯 Business Intelligence

- **5 Pre-defined Customer Segments** with characteristics
- **Marketing Strategies** for each segment
- **Budget Allocation Recommendations**
- **Expected ROI Calculations**
- **Engagement Scores**

### 🔧 Data Processing

- **Multiple File Format Support**

  - CSV
  - Excel (.xlsx, .xls)
  - ZIP archives
  - SQLite databases

- **Robust Data Pipeline**
  - Duplicate removal
  - Missing value handling
  - Outlier detection (Isolation Forest)
  - Feature scaling & transformation

---

## 🎬 Demo

### Dashboard Preview

```
📊 Customer Base Overview
├── Total Customers: 4,372
├── Avg Customer Value: £1,889
├── Avg Purchase Frequency: 4.8x
└── Total Revenue: £8,260,387
```

### Segment Distribution

| Segment                | Count | Avg Monetary | Revenue Share | Strategy           |
| ---------------------- | ----- | ------------ | ------------- | ------------------ |
| 🌟 VIP Champions       | 873   | £12,040      | 42%           | Retention & Reward |
| ✨ Potential Loyalists | 1,246 | £2,890       | 28%           | Development        |
| 📊 Average Customers   | 1,098 | £1,700       | 18%           | Activation         |
| ⚠️ At Risk             | 789   | £374         | 8%            | Re-engagement      |
| ❌ Lost/Hibernating    | 366   | £170         | 4%            | Last Resort        |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                          │
│         CSV / Excel / ZIP / SQL Database                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                Data Ingestion Layer                      │
│          ├─ ZipDataIngestor                             │
│          └─ DataIngestorFactory                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Data Preprocessing Layer                    │
│          ├─ Drop Duplicates & Missing Values            │
│          ├─ Data Type Conversion                        │
│          └─ Column Cleaning                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Feature Engineering Layer                     │
│          ├─ RFM Features Extraction                     │
│          ├─ CLV Calculation                             │
│          ├─ Basket Size Analysis                        │
│          ├─ Log Transformation                          │
│          ├─ Outlier Handling (Isolation Forest)         │
│          └─ Feature Scaling (Robust Scaler)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Model Training Layer                     │
│          ├─ K-Means Clustering                          │
│          ├─ Agglomerative Clustering                    │
│          ├─ Gaussian Mixture Model                      │
│          └─ Hyperparameter Tuning (Optuna)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Model Evaluation Layer                     │
│          ├─ Silhouette Score                            │
│          ├─ Davies-Bouldin Score                        │
│          ├─ Calinski-Harabasz Score                     │
│          ├─ Visual Evaluation (PCA)                     │
│          └─ Business Metrics Analysis                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Deployment Layer                         │
│          ├─ Streamlit Web Application                   │
│          ├─ Interactive Dashboards                      │
│          ├─ Marketing Strategy Engine                   │
│          └─ Export & Reporting Tools                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/RedDragon30/AI-Powered-Customer-Segmentation-System.git
cd customer-segmentation-system
```

### Step 2: Setup Your Environment

```bash
# Windows
./setup.bat

# Linux/Mac
# Grant execution permissions
chmod +x setup.sh

# Setup
./setup.sh
```

---

## 📖 Usage

### Option 1: Run Streamlit Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Option 2: Run Training Pipeline (Python Script)

```python
from steps.training_pipeline import training_pipeline
from steps.configs import ModelConfig

# Configure your model
config = ModelConfig()
config.file_path = "path/to/your/data.zip"
config.model_name = "k_means"  # Options: k_means, agg, gmm
config.fine_tuning = True  # Enable hyperparameter tuning

# Run the pipeline
model = training_pipeline()
```

### Option 3: Use Individual Components

```python
# Data Ingestion
from steps.data_ingestion import data_ingestion_step
df = data_ingestion_step("data/Online+Retail.zip")

# Data Cleaning
from steps.data_cleaning import data_cleaning_step
clean_df = data_cleaning_step(df)

# Feature Engineering
from steps.feature_engineering_step import feature_engineering_step
engineered_df, original_df = feature_engineering_step(clean_df)

# Model Building
from steps.model_building import model_building_step
from steps.configs import ModelConfig
config = ModelConfig()
model, predictions = model_building_step(engineered_df, config)

# Model Evaluation
from steps.model_evaluation import model_evaluation_step
metrics = model_evaluation_step(engineered_df, original_df, predictions)
```

---

## 📁 Project Structure

```
customer-segmentation-system/
│
├── 📂 src/                          # Source code modules
│   ├── ingest_data.py              # Data ingestion classes
│   ├── clean_data.py               # Data cleaning & preprocessing
│   ├── extract_features.py         # RFM feature extraction
│   ├── feature_engineering.py      # Feature transformation & scaling
│   ├── build_model.py              # ML model implementations
│   └── evaluate_model.py           # Model evaluation strategies
│
├── 📂 steps/                        # Pipeline steps
│   ├── data_ingestion.py           # Data loading step
│   ├── data_cleaning.py            # Cleaning step
│   ├── feature_engineering_step.py # Feature engineering step
│   ├── model_building.py           # Model training step
│   ├── model_evaluation.py         # Evaluation step
│   └── configs.py                  # Configuration settings
│
├── 📂 tests/                        # Test modules
│   └── get_test_data.py            # Test data generation
│
├── 📂 pipeline/
|│   └── training_pipeline.py        # Complete pipeline orchestration
|
├── 📂 artifacts/                    # Saved models & scalers
│   ├── k_means_model.pkl           # Trained model
│   └── robust_scaler.pkl           # Fitted scaler
│
├── 📂 data/                         # Data directory
│   └── Online+Retail.zip           # Sample dataset
│
├── 📂 demo/                         # Demo assets
│   └── customer_segments.png   # Demo image
|   ├── customer_segmentation.jpg
│
├── 📄 app.py                        # Streamlit web application
├── 📄 run_pipeline.py               # Run the training pipeline
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # Project documentation
└── 📄 LICENSE                       # License file
```

---

## 🤖 Models & Algorithms

### 1. K-Means Clustering

**Description**: Partitions customers into K clusters by minimizing within-cluster variance.

**Hyperparameters**:

- `n_clusters`: 3-5
- `init`: k-means++, random
- `max_iter`: 100-1000
- `algorithm`: lloyd, elkan

**Use Case**: Fast and efficient for large datasets with spherical clusters.

### 2. Agglomerative Clustering

**Description**: Hierarchical clustering that builds nested clusters by merging or splitting successively.

**Hyperparameters**:

- `n_clusters`: 3-5
- `linkage`: ward, complete, average, single
- `compute_distances`: True/False

**Use Case**: Better for non-spherical clusters and when hierarchy matters.

### 3. Gaussian Mixture Model (GMM)

**Description**: Probabilistic model that assumes data is generated from a mixture of Gaussian distributions.

**Hyperparameters**:

- `n_components`: 3-5
- `covariance_type`: full, tied, diag, spherical
- `max_iter`: 100-500

**Use Case**: Best for overlapping clusters and probabilistic assignments.

### Evaluation Metrics

| Metric                  | Description                                | Ideal Value      |
| ----------------------- | ------------------------------------------ | ---------------- |
| Silhouette Score        | Measures cluster cohesion and separation   | Close to 1       |
| Davies-Bouldin Score    | Average similarity ratio of clusters       | Close to 0       |
| Calinski-Harabasz Score | Ratio of between/within cluster dispersion | Higher is better |

---

## 🎯 Marketing Strategies

### 🌟 VIP Champions (Segment 2)

**Characteristics**: High frequency (14+ purchases), High monetary (£9,000+), Low recency (<20 days)

**Strategy**: Retention & Reward

- Exclusive VIP program with special privileges
- Personalized gifts for special occasions
- Direct personal contact
- Early access to new products
- Exclusive discounts 15-25%

**Budget**: 30-40% of marketing budget  
**ROI**: 🚀 Very High

### ✨ Potential Loyalists (Segment 3)

**Characteristics**: Good frequency (4-5 purchases), Average monetary (£2,000), Recent activity (<35 days)

**Strategy**: Development & Cross-selling

- Targeted promotional offers
- Progressive loyalty program
- Smart product recommendations
- Product bundle discounts

**Budget**: 25-30% of marketing budget  
**ROI**: 📈 High

### 📊 Average Customers (Segment 0)

**Characteristics**: Low frequency (2-3 purchases), Average monetary (£1,700), Average recency (85 days)

**Strategy**: Activation & Motivation

- Incentive offers for repeat purchases
- Targeted SMS/WhatsApp campaigns
- Limited-time discount coupons (10-15%)
- Interactive competitions

**Budget**: 20-25% of marketing budget  
**ROI**: 📊 Average

### ⚠️ At Risk (Segment 1)

**Characteristics**: Very low frequency (1-2 purchases), Low monetary (£374), Old purchase (145 days)

**Strategy**: Re-engagement & Retrieval

- "We Miss You" campaigns
- Huge discounts (20-30%)
- Personal calls for feedback
- Customer satisfaction surveys

**Budget**: 10-15% of marketing budget  
**ROI**: 📉 Medium-Low

### ❌ Lost/Hibernating (Segment 4)

**Characteristics**: Very low frequency (1 purchase), Low monetary (£170), Very old purchase (163+ days)

**Strategy**: Last Resort

- "Last Chance" campaigns
- Huge discounts (40-50%)
- Exit surveys
- Evaluate if worth investment

**Budget**: 5-10% of marketing budget  
**ROI**: ⚠️ Low

---

## 📊 Sample Data Format

Your data should include the following columns:

| Column      | Type       | Description                |
| ----------- | ---------- | -------------------------- |
| CustomerID  | string/int | Unique customer identifier |
| InvoiceID   | string     | Unique invoice identifier  |
| ProductID   | string     | Unique product identifier  |
| InvoiceDate | datetime   | Date of purchase           |
| Quantity    | int        | Quantity purchased         |
| UnitPrice   | float      | Price per unit             |

**Optional columns**: Description, Country

---

## 🧪 Testing

Run tests to ensure everything works:

```bash
# Test data creation
python tests/get_test_data.py

# Run complete pipeline
python steps/training_pipeline.py
```

---

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Contact

**Emran Albeik**  
ML Engineer | Data Analyst

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](http://linkedin.com/in/emranalbeik)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/RedDragon30)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:emranalbiek@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://emranalbeik.odoo.com/)

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Online Retail Dataset
- Streamlit team for the amazing framework
- scikit-learn community for ML tools
- Optuna team for hyperparameter optimization

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by Emran Albeik

</div>
