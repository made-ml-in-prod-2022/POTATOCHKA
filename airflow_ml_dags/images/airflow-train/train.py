import os
import pandas as pd
import click
import pickle
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def train(data_dir: str, artifacts_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_data = pd.read_csv(os.path.join(data_dir, "data_train.csv"))
    train_target = pd.read_csv(os.path.join(data_dir, "target_train.csv"))

    scaler = pickle.load(open(artifacts_dir + "/transform.pkl", 'rb'))
    scaled_data = scaler.transform(train_data)

    model = LogisticRegression()
    model.fit(scaled_data, train_target)

    pickle.dump(model, open(output_dir + "/model.pkl", 'wb'))

if __name__ == '__main__':
    train()
