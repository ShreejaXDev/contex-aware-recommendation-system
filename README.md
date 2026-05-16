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
context-aware-recommendation-system/
├── data/
│   ├── raw/              # Raw input data
│   ├── processed/        # Processed/cleaned data
│   └── sample/           # Sample data for testing
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── preprocessing/    # Data preprocessing modules
│   ├── feature_engineering/  # Feature extraction and engineering
│   ├── training/         # Model training scripts
│   ├── evaluation/       # Model evaluation metrics
│   ├── api/             # FastAPI application
│   ├── redis_cache/     # Redis caching utilities
├── models/              # Trained model artifacts
├── airflow/             # Airflow DAG definitions
├── tests/               # Unit and integration tests
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── main.py             # Application entry point
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

## Usage

### Run the API Server
```bash
python main.py
```

The API will be available at `http://localhost:8000`

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

## Contributors

List of project contributors

## License

MIT License

## Contact

For questions or support, please open an issue or contact the development team.

