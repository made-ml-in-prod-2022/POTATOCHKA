import os
import pandas as pd
import click
import pickle
import json

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@click.command("validate")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def validate(data_dir: str, artifacts_dir: str, output_dir: str):
    test_data = pd.read_csv(os.path.join(data_dir, "data_test.csv"))
    test_target = pd.read_csv(os.path.join(data_dir, "target_test.csv"))

    scaler = pickle.load(open(artifacts_dir + "/transform.pkl", 'rb'))
    scaled_data = scaler.transform(test_data)

    model = pickle.load(open(artifacts_dir + "/model.pkl", 'rb'))
    y_pred = model.predict(scaled_data)
    y_proba = model.predict_proba(scaled_data)[:, 1]

    metrics = {'accuracy': accuracy_score(test_target, y_pred),
               "auc": roc_auc_score(test_target, y_proba),
               "f1": f1_score(test_target, y_pred)}

    with open(output_dir + "/metrics.pkl", "w") as fout:
        json.dump(metrics, fout)


if __name__ == '__main__':
    validate()
