# Context-Aware Recommendation System


A production-grade machine learning recommendation system that leverages context-aware algorithms and Redis caching for efficient real-time recommendations.

## Features

- **Context-Aware Recommendations**: Generates recommendations based on user context and item features
- **BruteForce Retrieval**: TensorFlow Recommenders BruteForce top-k retrieval
- **Redis Caching**: High-performance caching for frequently accessed recommendations
- **REST API**: FastAPI-based API for serving recommendations
- **Apache Airflow**: Orchestrated data pipelines for model training and evaluation
- **Scalable Architecture**: Designed for production deployment

## Project Structure

```
contex-aware-recommendation-system/
├── airflow/                 # Airflow DAG definitions
├── data/
│   ├── raw/                  # Raw input data
│   ├── processed/            # Processed/cleaned data
│   └── sample/               # Sample data for testing
├── docs/                     # Architecture and design docs
│   └── tests/                # Test documentation
├── frontend/                 # Frontend app
├── notebooks/                # Jupyter notebooks for exploration
├── saved_models/             # Saved model weights and configs
├── src/
│   ├── api/                  # FastAPI application
│   ├── evaluation/           # Model evaluation scripts
│   ├── feature_engineering/  # Feature extraction and engineering
│   ├── models/               # Model definitions and inference
│   ├── preprocessing/        # Data preprocessing modules
│   ├── redis_cache/          # Redis caching utilities
│   └── training/             # Model training scripts
├── tests/                    # Unit and integration tests
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── main.py                   # Application entry point
```

## Installation

### Prerequisites
- Python 3.10+
- Redis (for caching)
- MongoDB/PostgreSQL (for data storage)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd context-aware-recommendation-system
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate   # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## End-to-End Setup (From Clone to Working Project)

1. Clone the repository and enter the folder:
```bash
git clone <repository-url>
cd contex-aware-recommendation-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate   # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset and place raw files under:
```
data/raw/
```
Expected raw files include (example):
- articles.csv
- customers.csv
- transactions_train.csv

5. Run preprocessing to create cleaned files under:
```
data/processed/
```
You can use the scripts in `src/preprocessing/` to generate:
- articles_cleaned.csv
- customers_cleaned.csv
- transactions_cleaned.csv

6. Run feature engineering to create feature files under:
```
data/processed/
```
Use scripts in `src/feature_engineering/` to produce:
- interaction_features.csv
- item_features.csv
- user_features.csv

7. Train the model (details below). The trained weights and configs are saved in:
```
saved_models/
```

## Usage

### Run the API Server (Backend)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Run the Frontend (Streamlit)
```bash
streamlit run frontend/app.py
```

The Streamlit app will be available at `http://localhost:8501`

### Train the Model
```bash
python src/models/train_model.py
```

### Generate Recommendations (Training-Level Inference)
Use this if you want to generate recommendation outputs after training:
```bash
python src/models/generate_recommendations.py
```

### Run Data Pipeline
```bash
python -m airflow webserver
python -m airflow scheduler
```

### Run Tests
```bash
pytest tests/ -v --cov=src
```

## Configuration

Configuration is managed through environment variables. See `.env.example` for details.

## API Endpoints

- `POST /recommend` - Get recommendations for a user
- `GET /health` - Health check
- `POST /cache/clear` - Clear recommendation cache



## License

MIT License

## Contact

For questions or support, please open an issue or contact the development team.

