import os
import click
import pandas as pd
from sklearn.datasets import make_classification


def generate_data(output_dir, size=10000):
    feats, target = make_classification(n_samples=size, n_features=4, n_informative=2, n_classes=2)
    pd.DataFrame(feats).to_csv(os.path.join(output_dir, "data.csv"), index=False)
    pd.DataFrame(target).to_csv(os.path.join(output_dir, "target.csv"), index=False)


@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    generate_data(output_dir)

if __name__ == '__main__':
    download()
