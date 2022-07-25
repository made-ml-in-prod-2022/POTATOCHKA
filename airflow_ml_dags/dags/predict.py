import airflow

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from constants import DEFAULT_ARGS, PATH_TO_VOLUME, PATH_TO_RAW, PATH_TO_ARTIFACTS, PATH_TO_PREDICTS

with DAG(
        dag_id="predict",
        start_date=airflow.utils.dates.days_ago(5),
        schedule_interval="@daily",
        default_args=DEFAULT_ARGS,
) as dag:
    wait_model = FileSensor(
        task_id="wait-for-model",
        filepath="model_artifacts/{{ ds }}/model.pkl",
        fs_conn_id="MY_CONN",
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke")

    wait_transformer = FileSensor(
        task_id="wait-for-transformer",
        filepath="model_artifacts/{{ ds }}/transform.pkl",
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

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--data-dir={PATH_TO_RAW} --artifacts-dir={PATH_TO_ARTIFACTS} --output-dir={PATH_TO_PREDICTS}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[PATH_TO_VOLUME]
    )

    [wait_model, wait_transformer, wait_data] >> predict
