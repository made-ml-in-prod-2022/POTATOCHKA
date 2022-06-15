import hydra
import sys
import pandas as pd
import pickle
import logging
from typing import Dict

sys.path.append('./utils/')
from read_config import create_config
from transform_class import DatasetTransformer
from train_model import train_model
from split import split_and_save_data


def train_pipeline(cfg: Dict[str, str]) -> Dict[str, float]:
    dataclass_config = create_config(cfg)
    data_path = hydra.utils.to_absolute_path(path=dataclass_config.input_data_path)
    logging.info(f"read data from {data_path} for training")
    full_df = pd.read_csv(data_path)
    target = full_df[dataclass_config.feature_params.target_col]
    full_df.drop([dataclass_config.feature_params.target_col], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = split_and_save_data(full_df, target, dataclass_config)

    transform = DatasetTransformer(dataclass_config.feature_params.transforms)
    transform.fit(X_train)

    path_to_transform = hydra.utils.to_absolute_path(path=dataclass_config.path_to_transform)
    pickle.dump(transform, open(path_to_transform, 'wb'))

    X_train_transformed = transform.transform(X_train)
    X_test_transformed = transform.transform(X_test)

    metrics = train_model(dataclass_config, X_train_transformed, X_test_transformed, y_train, y_test)
    return metrics


@hydra.main(config_path="../configs")
def main(cfg):
    train_pipeline(cfg)


if __name__ == "__main__":
    main()
