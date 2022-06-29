import uvicorn
import sys
import pandas as pd
from fastapi import FastAPI
import pickle
from typing import List, Union
import logging
from config import InputToPredict, check_input, HOST, PORT, PATH_TO_MODEL, PATH_TO_TRANSFORMER
from ml_project.utils import transform_class


logging.basicConfig(filename='server.log', filemode='w', level=logging.INFO)

app = FastAPI()

model = None
transformer = None


@app.get("/health")
def health() -> bool:
    return not (model is None) and not (transformer is None)


def make_predict(data: List[List[float]], feature_names: List[str]) -> dict:
    df = pd.DataFrame(data, columns=feature_names)
    X_transformed = transformer.transform(df)
    logging.info(X_transformed)
    predicts = model.predict_proba(X_transformed)[:, 1]
    logging.info("predictions are calculated")
    return {'probability': list(predicts)}


@app.get("/predict", response_model=dict)
def predict(request: InputToPredict):
    assert model is not None
    check_input(request)
    logging.info("data validation successful")
    logging.info("make prediction")
    return make_predict(request.data, request.features)


@app.on_event("startup")
def prepare_model():
    global model, transformer
    model_path = PATH_TO_MODEL
    path_to_transform = PATH_TO_TRANSFORMER
    if model_path is None:
        err = f"model_path is None"
        raise RuntimeError(err)
    if path_to_transform is None:
        err = f"path_to_transform is None"
        raise RuntimeError(err)
    try:
        model = pickle.load(open(model_path, 'rb'))
        transformer = pickle.load(open(path_to_transform, 'rb'))
        logging.info("model and transformer loaded")
    except FileNotFoundError:
        logging.error('no model')
        sys.exit()


@app.get("/")
def main():
    logging.info("start server")
    return "Start online model inference"


if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
