import logging
from dataclasses import dataclass, field
from typing import List, Dict

import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class SplitParams:
    test_size: float = field(default=0.2)
    seed: int = field(default=42)


@dataclass()
class TrainParams:
    model_type: str = field(default="LogisticRegression")
    params: dict = field(default_factory=dict)


@dataclass()
class ColumnTransformerParams:
    transform: str
    columns: List[str]


@dataclass()
class FeatureParams:
    transforms: List[ColumnTransformerParams]
    target_col: str


@dataclass()
class Config:
    input_data_path: str
    experiment_path: str
    output_model_fname: str
    fitted_transform_fname: str
    metric_fname: str
    train_data_path: str
    test_data_path: str
    test_label_path: str
    predict_path: str
    split_params: SplitParams
    train_params: TrainParams
    feature_params: FeatureParams


ConfigSchema = class_schema(Config)


def create_config(config_file: Dict[str, str]) -> Config:
    if config_file is str:
        with open(config_file) as f:
            logging.info(f"building config from {config_file}")
            config_dict = yaml.safe_load(f)
    else:
        logging.info("building config from dict")
        config_dict = config_file
    schema = ConfigSchema()

    config_dataclass = schema.load(config_dict)
    return config_dataclass
