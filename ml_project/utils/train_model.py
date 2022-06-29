import logging
import hydra
import pandas as pd
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle
import json
import sys
from typing import Dict

sys.path.append('./utils/')
from read_config import Config


def train_model(dataclass_config: Config, X_train_transformed: pd.DataFrame, X_test_transformed: pd.DataFrame,
                y_train: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
    if dataclass_config.train_params.model_type == "LogisticRegression":
        logging.info(f'train logreg with params:{dataclass_config.train_params.params}')
        model = LogisticRegression(**dataclass_config.train_params.params)
        model.fit(X_train_transformed, y_train)
    elif dataclass_config.train_params.model_type == "Boosting":
        logging.info(f'train boosting with params:{dataclass_config.train_params.params}')
        model = CatBoostClassifier(**dataclass_config.train_params.params)
        model.fit(X_train_transformed, y_train, verbose=False)
    else:
        logging.info(f'no valid models in input file, train logreg')
        model = LogisticRegression(**dataclass_config.train_params.params)
        model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[::, 1]
    logging.info('calc metrics')

    metrics = {'accuracy': accuracy_score(y_test, y_pred),
               "auc": roc_auc_score(y_test, y_pred_proba),
               "f1": f1_score(y_test, y_pred)}

    path_to_model = hydra.utils.to_absolute_path(path=dataclass_config.path_to_model)
    pickle.dump(model, open(path_to_model, 'wb'))
    path_to_save_metrics = dataclass_config.experiment_path + '/' + dataclass_config.metric_fname
    path_to_save_metrics = hydra.utils.to_absolute_path(path=path_to_save_metrics)
    with open(path_to_save_metrics, "w") as fout:
        logging.info(f'model saved here: {path_to_save_metrics}')
        json.dump(metrics, fout)
    return metrics
