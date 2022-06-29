from pydantic import BaseModel, conlist
from fastapi import HTTPException
from typing import Union
import pandas as pd
import numpy as np

HOST = "0.0.0.0"
PORT = 8080
PATH_TO_MODEL = "ml_project/models/logreg/model.pkl"
PATH_TO_TRANSFORMER = "ml_project/models/logreg/transform.pkl"

COLUMNS = ['id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
           'thal']
BINARY_FEATS = ["sex", "fbs", "exang"]
CATEGORICAL_FEATS = {"names": ["cp", "restecg", "slope", "ca", "thal"],
                     "min_max": [[0, 3], [0, 2], [0, 2], [0, 3], [0, 2]]}
NUMERICAL_FEATS = {'names': ["age", "trestbps", "chol", "thalach", "oldpeak"],
                   "min_max": [[29, 77], [94, 200], [126, 564], [71, 202], [0, 6.2]]}


class InputToPredict(BaseModel):
    data: conlist(conlist(Union[float, int], min_items=14, max_items=14), min_items=1)
    features: conlist(str, min_items=14, max_items=14)


def check_input(request: InputToPredict) -> None:
    if not isinstance(request, InputToPredict):
        raise HTTPException(status_code=400, detail="incorrect input type")

    if len(request.data) == 0:
        raise HTTPException(status_code=400, detail="empty data")

    if request.features != COLUMNS:
        raise HTTPException(status_code=400, detail="mismatch in columns")

    if len(request.data[0]) != len(request.features):
        raise HTTPException(status_code=400, detail="mismatch in feature to data")

    data = pd.DataFrame(data=request.data, columns=request.features).iloc[0]
    if data.isna().sum() != 0:
        raise HTTPException(status_code=400, detail="Nans in input")

    if not np.array_equal(data.values, data.values.astype(float)):
        raise HTTPException(status_code=400, detail=f"non numerical data in input")

    for idx, feat_name in enumerate(CATEGORICAL_FEATS["names"]):
        min_val = CATEGORICAL_FEATS["min_max"][idx][0]
        max_val = CATEGORICAL_FEATS["min_max"][idx][1]
        if data[feat_name] < min_val or data[feat_name] > max_val:
            raise HTTPException(status_code=400, detail=f"incorrect value {data[feat_name]} in {feat_name}")

    for feat in BINARY_FEATS:
        if data[feat] < 0 or data[feat] > 1:
            raise HTTPException(status_code=400, detail=f"incorrect value {data[feat]} in {feat}")
