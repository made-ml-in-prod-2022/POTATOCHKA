import sys
import numpy as np
import pandas as pd
from os.path import exists

sys.path.append('./utils/')
from transform_class import DatasetTransformer
from read_config import create_config
from transform_class import DatasetTransformer


def test_dataclass_transformer():
    # file_exists = exists('./configs/test_cfg.yaml')
    dataclass_config = create_config('./configs/test_cfg.yaml')
    data = np.hstack((np.ones((100, 1)), np.random.random((100, 1)) + 100))
    df = pd.DataFrame(data=data, columns=['cat_feat', 'num_feat'])
    df.iloc[0, 0] = 0

    transform = DatasetTransformer(dataclass_config.feature_params.transforms)
    transform.fit(df)
    df_transformed = transform.transform(df)
    assert df_transformed.shape == (100, 3)

    assert np.isclose(df_transformed[:, 0].std(), 1, atol=1e-3)  # проверка стандарт скалера
    assert np.isclose(df_transformed[:, 0].mean(), 0, atol=1e-3)
    assert df_transformed[0, 1] == 1 # проверка one hot encoder
    assert df_transformed[0, 2] == 0
