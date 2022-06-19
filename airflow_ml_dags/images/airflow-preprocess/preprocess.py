import os
import pandas as pd
import click
from sklearn.preprocessing import MinMaxScaler
import pickle

@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    train_data = pd.read_csv(os.path.join(input_dir, "data_train.csv"))
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(scaler, open(output_dir + "/transform.pkl", 'wb'))


if __name__ == '__main__':
    preprocess()
