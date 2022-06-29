import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np



class DatasetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transform_to_features):
        self.transform_to_features = transform_to_features
        self.ohe = OneHotEncoder()
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        logging.info(f"fit transformer with params: {self.transform_to_features}")
        for transform_part in self.transform_to_features:
            transform = transform_part.transform
            columns = transform_part.columns
            if transform == 'StandardScaler':
                self.scaler.fit(X[columns])
            elif transform == 'OneHotEncoder':
                self.ohe.fit(X[columns])
        return self

    def transform(self, X, y=None):
        logging.info(f"transformer transform")
        transformed_features = []
        for transform_part in self.transform_to_features:
            transform = transform_part.transform
            columns = transform_part.columns
            if transform == 'StandardScaler':
                scaled_feats = self.scaler.transform(X[columns])
                transformed_features.append(scaled_feats)
            elif transform == 'OneHotEncoder':
                ohe_feats = self.ohe.transform(X[columns]).toarray()
                transformed_features.append(ohe_feats)
            elif transform == 'pass':
                default_feats = X[columns]
                transformed_features.append(default_feats)
        X_transformed = np.hstack(transformed_features)
        return X_transformed