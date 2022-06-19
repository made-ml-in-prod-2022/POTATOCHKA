import os
import pandas as pd
import pickle
import click


@click.command("predict")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def predict(data_dir: str, artifacts_dir: str, output_dir: str):
    print(data_dir)
    data = pd.read_csv(os.path.join(data_dir, "data.csv"))

    scaler = pickle.load(open(artifacts_dir + "/transform.pkl", 'rb'))
    scaled_data = scaler.transform(data)

    model = pickle.load(open(artifacts_dir + "/model.pkl", 'rb'))
    y_proba = model.predict_proba(scaled_data)[:, 1]

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(y_proba).to_csv(os.path.join(output_dir, 'predicts.csv'), index=False, header=None)


if __name__ == '__main__':
    predict()
