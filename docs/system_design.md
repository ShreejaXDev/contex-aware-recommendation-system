# Context-Aware Neural Recommendation Engine

## Project Flow

H&M Dataset
↓
Data Preprocessing
↓
Feature Engineering
↓
Two-Tower Recommendation Model
↓
Embedding Generation
↓
ANN Retrieval
↓
Redis Feature Store
↓
FastAPI Recommendation API
↓
Personalized Product Recommendations

---

## Main Components

### 1. Data Processing
- Clean datasets
- Handle missing values
- Generate sample datasets

### 2. Feature Engineering
- User purchase history
- Product popularity
- Recency features

### 3. Recommendation Model
- TensorFlow Recommenders
- Two-Tower Architecture
- User & Item embeddings

### 4. Retrieval System
- ANN search using FAISS
- Fast candidate retrieval

### 5. Backend API
- FastAPI recommendation endpoints

### 6. Pipeline Automation
- Airflow DAGs for retraining