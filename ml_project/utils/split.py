import hydra
import numpy as np
import pandas as pd
import os
import logging
from typing import Tuple
import sys
from sklearn.model_selection import train_test_split

sys.path.append('./utils/')
from read_config import Config


def split_and_save_data(full_df: pd.DataFrame, target: pd.DataFrame,
                        dataclass_config: Config) -> Tuple[np.array]:
    if not os.path.exists(hydra.utils.to_absolute_path(path=dataclass_config.experiment_path)):
        logging.info(f"create experiment dir:{hydra.utils.to_absolute_path(path=dataclass_config.experiment_path)}")
        os.makedirs(hydra.utils.to_absolute_path(path=dataclass_config.experiment_path))
    logging.info(f"split dataset with {dataclass_config.split_params}")
    X_train, X_test, y_train, y_test = train_test_split(
        full_df,
        target,
        test_size=dataclass_config.split_params.test_size,
        random_state=dataclass_config.split_params.seed)
    logging.info(f"save train and test parts")
    path_to_save_train_part = hydra.utils.to_absolute_path(path=dataclass_config.train_data_path)
    path_to_save_test_part = hydra.utils.to_absolute_path(path=dataclass_config.test_data_path)
    path_to_save_test_part_label = hydra.utils.to_absolute_path(path=dataclass_config.test_label_path)
    pd.concat([X_train, y_train], axis=1).to_csv(path_or_buf=path_to_save_train_part, index=False)
    X_test.to_csv(path_or_buf=path_to_save_test_part, index=False)
    target.to_csv(path_or_buf=path_to_save_test_part_label, index=False)
    return X_train, X_test, y_train, y_test
