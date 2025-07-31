# MLops Project
Airflow, Spark, MinIO, FastAPI, Pytorch를 활용한 머신러닝 파이프라인 프로젝트입니다.
작성한 순서대로 개발을 진행할 예정입니다.

## Project Structure

```
MLops/
├── airflow/                # Airflow workflow management
│   ├── dags/              # DAG files
│   ├── plugins/           # Custom plugins
│   └── config/            # Configuration files
├── spark/                 # Spark cluster and jobs
│   ├── spark-jobs/        # Spark job codes
│   └── shared/            # Shared resources
└── train/                 # ML training data and models
    └── data/              # Training data (HDF5, Parquet)
```

## Key Components

### Airflow
- **polars_minio_ml_pipeline**: MongoDB → Polars → MinIO(Parquet) → ML Training pipeline
- Data lake construction using MinIO
- High-performance data processing with Polars

### Spark
- Spark cluster for distributed data processing
- Large-scale data transformation and aggregation

### Train
- Data storage for machine learning model training
- Support for HDF5 and Parquet formats

## Getting Started

1. Run Airflow:
```bash
cd airflow
docker-compose up -d
```

2. Run Spark cluster:
```bash
cd spark
docker-compose -f docker-compose.yaml up -d
```

## Data Pipeline

1. **Data Collection**: Extract raw data from MongoDB
2. **Data Processing**: High-speed data transformation using Polars
3. **Data Storage**: Store in Parquet format in MinIO
4. **Model Training**: Train ML models with preprocessed data