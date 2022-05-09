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
    logging.info('Start training')
    dataclass_config = create_config(cfg)
    data_path = hydra.utils.to_absolute_path(path=dataclass_config.input_data_path)
    logging.info(f"read data from {data_path}")
    full_df = pd.read_csv(data_path)
    target = full_df[dataclass_config.feature_params.target_col]
    full_df.drop([dataclass_config.feature_params.target_col], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = split_and_save_data(full_df, target, dataclass_config)

    transform = DatasetTransformer(dataclass_config.feature_params.transforms)
    transform.fit(X_train)

    path_to_transform = hydra.utils.to_absolute_path(
        path=dataclass_config.experiment_path + '/' + dataclass_config.fitted_transform_fname)
    pickle.dump(transform, open(path_to_transform, 'wb'))

    logging.info('transform train')
    X_train_transformed = transform.transform(X_train)
    logging.info('transform test')
    X_test_transformed = transform.transform(X_test)

    metrics = train_model(dataclass_config, X_train_transformed, X_test_transformed, y_train, y_test)
    logging.info('End of training')
    return metrics


@hydra.main(config_path="../configs")
def main(cfg):
    train_pipeline(cfg)


if __name__ == "__main__":
    main()
