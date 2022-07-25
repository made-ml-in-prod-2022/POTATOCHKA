import airflow
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import DEFAULT_ARGS, PATH_TO_VOLUME

with DAG(
        dag_id="generate_data",
        start_date=airflow.utils.dates.days_ago(5),
        schedule_interval="@daily",
        default_args=DEFAULT_ARGS,
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=[PATH_TO_VOLUME]
    )

    download
