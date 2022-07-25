import airflow

from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from constants import DEFAULT_ARGS, PATH_TO_VOLUME, PATH_TO_RAW, PATH_TO_PROCESSED, PATH_TO_ARTIFACTS

with DAG(
        dag_id="train",
        start_date=airflow.utils.dates.days_ago(5),
        schedule_interval="@daily",
        default_args=DEFAULT_ARGS,
) as dag:

    wait_target = FileSensor(
        task_id="wait-for-target",
        filepath="raw/{{ ds }}/target.csv",
        fs_conn_id="MY_CONN",
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke")

    wait_data = FileSensor(
        task_id="wait-for-data",
        filepath="raw/{{ ds }}/data.csv",
        fs_conn_id="MY_CONN",
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke")

    split_data = DockerOperator(
        image="airflow-split",
        command=f"--input-dir={PATH_TO_RAW} --output-dir={PATH_TO_PROCESSED}",
        network_mode="bridge",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[PATH_TO_VOLUME]
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir={PATH_TO_PROCESSED} --output-dir={PATH_TO_ARTIFACTS}",
        network_mode="bridge",
        task_id="docker-airflow-fit-scaler",
        do_xcom_push=False,
        volumes=[PATH_TO_VOLUME]
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--data-dir={PATH_TO_PROCESSED} --artifacts-dir={PATH_TO_ARTIFACTS} --output-dir={PATH_TO_ARTIFACTS}",
        network_mode="bridge",
        task_id="docker-airflow-fit-model",
        do_xcom_push=False,
        volumes=[PATH_TO_VOLUME]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--data-dir={PATH_TO_PROCESSED} --artifacts-dir={PATH_TO_ARTIFACTS} --output-dir={PATH_TO_ARTIFACTS}",
        network_mode="bridge",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=[PATH_TO_VOLUME]
    )

    [wait_target, wait_data] >> split_data >> preprocess >> train >> validate
