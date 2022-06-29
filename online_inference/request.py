import logging
import pandas as pd
import requests
from config import HOST, PORT

logging.basicConfig(filename='request.log', filemode='w', level=logging.INFO)
dataframe_path = 'ml_project/data/raw/test_data.csv'


def main():
    df = pd.read_csv(dataframe_path)
    logging.info(f"read df from {dataframe_path}")
    features = ['id'] + list(df.columns)
    for row in df.itertuples():
        data = [x for x in row]
        request = {"data": [data], "features": features}
        response = requests.get(
            f"http://{HOST}:{PORT}/predict",
            json=request,
        )
        logging.info(f"status_code: {response.status_code}")
        logging.info(f"response.json:: {response.json()}")


if __name__ == "__main__":
    main()
