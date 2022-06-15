import hydra
import sys
import pandas as pd
import pickle
import logging
from typing import Dict

sys.path.append('./utils/')
from read_config import create_config


def predict(cfg: Dict[str, str]) -> None:
    dataclass_config = create_config(cfg)
    logging.info(f"read data from {hydra.utils.to_absolute_path(path=dataclass_config.test_data_path)}")
    df = pd.read_csv(hydra.utils.to_absolute_path(path=dataclass_config.test_data_path))

    path_to_transform = hydra.utils.to_absolute_path(path=dataclass_config.path_to_transform)
    logging.info(f'Load transformer from:{path_to_transform}')

    transform = pickle.load(open(path_to_transform, 'rb'))
    X_transformed = transform.transform(df)
    path_to_model = hydra.utils.to_absolute_path(path=dataclass_config.path_to_model)
    logging.info(f'Load model from:{path_to_model}')
    model = pickle.load(open(path_to_model, 'rb'))
    logging.info(f'Model predict and save predicts:{hydra.utils.to_absolute_path(path=dataclass_config.predict_path)}')
    predicts = model.predict_proba(X_transformed)[:, 1]
    predicts = pd.DataFrame(predicts, columns=["target"])
    predicts.to_csv(hydra.utils.to_absolute_path(path=dataclass_config.predict_path), index=False)


@hydra.main(config_path="../configs/")
def main(cfg):
    predict(cfg)


if __name__ == "__main__":
    main()
