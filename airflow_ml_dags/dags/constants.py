from datetime import timedelta


DEFAULT_ARGS = {
    "owner": "admin",
    "email": ["admin@example.com"],
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
    'email_on_failure': True
}

PATH_TO_VOLUME = "/home/danil/notebook/simple_projects/MADE/ML_In_Prod/Homeworks/airflow_ml_dags/data:/data"
PATH_TO_RAW = "/data/raw/{{ ds }}"
PATH_TO_PROCESSED = "/data/processed/{{ ds }}"
PATH_TO_ARTIFACTS = "/data/model_artifacts/{{ ds }}"
PATH_TO_PREDICTS = "/data/predictions/{{ ds }}"

